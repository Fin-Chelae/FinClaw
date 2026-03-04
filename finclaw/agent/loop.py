"""Agent loop: the core processing engine."""

import asyncio
import os
from contextlib import AsyncExitStack
import json
import json_repair
from pathlib import Path
from typing import Any

from loguru import logger

from finclaw.bus.events import InboundMessage, OutboundMessage
from finclaw.bus.queue import MessageBus
from finclaw.providers.base import LLMProvider
from finclaw.agent.context import ContextBuilder
from finclaw.agent.tools.registry import ToolRegistry
from finclaw.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from finclaw.agent.tools.shell import ExecTool
from finclaw.agent.tools.web import TavilySearchTool, BraveSearchTool, WebFetchTool
from finclaw.agent.tools.message import MessageTool
from finclaw.agent.tools.spawn import SpawnTool
from finclaw.agent.tools.cron import CronTool
from finclaw.agent.memory import MemoryStore
from finclaw.agent.subagent import SubagentManager
from finclaw.agent.financial.router import FinancialMetricsRouter, FinancialSearchRouter
from finclaw.agent.financial.equity_valuation_router import EquityValuationRouter
from finclaw.agent.financial.economics_router import EconomicsRouter
from finclaw.agent.financial_tools import EconomicsDataTool
from finclaw.agent.financial.meme_router import MemeRouter
from finclaw.agent.financial.prediction_market_router import PredictionMarketRouter
from finclaw.agent.financial_tools.financial_news_tool import FinancialNewsTool
from finclaw.session.manager import Session, SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        inner_model: str = "",
        inner_provider: LLMProvider | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        search_api_key: str | None = None,
        search_provider: str = "tavily",
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
    ):
        from finclaw.config.schema import ExecToolConfig
        from finclaw.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.inner_model = inner_model
        self.inner_provider = inner_provider
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.search_api_key = search_api_key
        self.search_provider = search_provider
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            search_api_key=search_api_key,
            search_provider=search_provider,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._register_default_tools()

        # Financial specialization modules
        from finclaw.agent.financial import (
            FinancialIntentDetector,
            FinanceProfileManager,
            FinancialHistoryManager,
            FinancialDataCache,
        )

        self.fin_intent = FinancialIntentDetector()
        self.fin_profile = FinanceProfileManager(workspace)
        self.fin_history = FinancialHistoryManager(
            history_path=workspace / "memory" / "FINANCIAL_HISTORY.md"
        )
        self.fin_cache = FinancialDataCache(
            cache_dir=workspace / "financial_data"
        )

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web search: register based on configured provider
        if self.search_provider == "brave" and (self.search_api_key or os.environ.get("BRAVE_API_KEY")):
            search_tool = BraveSearchTool(api_key=self.search_api_key)
        else:
            search_tool = TavilySearchTool(api_key=self.search_api_key)
        self.tools.register(search_tool)
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Financial tools — LLM sub-agent routers (Dexter pattern)
        _r = dict(
            provider=self.provider,
            model=self.model,
            inner_model=self.inner_model,
            inner_provider=self.inner_provider,
            workspace=self.workspace,
        )
        self.tools.register(FinancialMetricsRouter(**_r))
        self.tools.register(FinancialSearchRouter(**_r, search_tool=search_tool))
        self.tools.register(EquityValuationRouter(**_r))
        self.tools.register(EconomicsDataTool())
        self.tools.register(EconomicsRouter(**_r))

        # Meme coin router — LLM sub-agent (market data + social scanning + token creation)
        self.tools.register(MemeRouter(**_r))

        # Prediction market router — LLM sub-agent (Polymarket + Kalshi)
        self.tools.register(PredictionMarketRouter(**_r))

        # Financial news (Bloomberg, MarketWatch, Google News)
        self.tools.register(FinancialNewsTool())

    
    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from finclaw.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    async def _run_agent_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.

        Returns:
            Tuple of (final_content, list_of_tools_used).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"→ tool: {tool_call.name}  args: {args_str[:300]}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    result_preview = (result or "")[:200].replace("\n", " ")
                    logger.debug(f"← result: {tool_call.name}  {result_preview}")
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                messages.append({"role": "user", "content": "Based on the tool results, either call another tool if needed or provide your final answer to the user."})
            else:
                final_content = response.content
                break
        else:
            # Exhausted max_iterations — force a final text response without tools
            logger.warning(f"Agent loop hit max iterations ({self.max_iterations}), forcing final response")
            messages.append({
                "role": "user",
                "content": (
                    "You have used all available tool iterations. "
                    "Summarize what you accomplished and what remains, then respond to the user."
                ),
            })
            try:
                closing = await self.provider.chat(
                    messages=messages,
                    tools=[],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                final_content = closing.content
            except Exception as e:
                logger.error(f"Closing response failed: {e}")

        return final_content, tools_used

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        
        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="📊 FinClaw commands:\n/start — Introduction & profile setup\n/new — Start a new conversation\n/help — Show available commands")
        if cmd == "/start":
            # Clear session for a fresh start, then let LLM introduce + onboard
            session.clear()
            self.sessions.save(session)
            msg.content = (
                "The user just started the bot. Introduce yourself as FinClaw 📊, "
                "briefly explain your capabilities (stock/crypto prices, financial analysis, "
                "market news, portfolio tracking), then ask the investment profile questions "
                "to get to know them (2-3 questions at a time, keep it friendly and concise)."
            )

        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        original_user_content = msg.content

        # Financial pre-hook: intent detection, profile check, cache/history lookup
        intent = self.fin_intent.detect(msg.content)
        finance_context_parts: list[str] = []

        if intent.is_financial:
            if not self.fin_profile.exists():
                # Create a draft profile immediately to avoid repeated onboarding loops.
                self.fin_profile.bootstrap_from_message(msg.content, intent.tickers)
            if not self.fin_profile.is_complete():
                from finclaw.agent.financial.prompts import ONBOARDING_PROMPT
                onboarding_prompt = ONBOARDING_PROMPT.replace(
                    "__WORKSPACE_PATH__", str(self.workspace.expanduser().resolve())
                )
                finance_context_parts.append(onboarding_prompt)

            # Inject macro tool routing when intent is macro_analysis
            if intent.intent_type == "macro_analysis":
                from finclaw.agent.financial.prompts import MACRO_ROUTING_PROMPT
                finance_context_parts.append(MACRO_ROUTING_PROMPT)

            # Inject meme routing when intent is meme
            if intent.intent_type == "meme":
                from finclaw.agent.financial.prompts import MEME_ROUTING_PROMPT
                finance_context_parts.append(MEME_ROUTING_PROMPT)

            # Inject prediction market routing when intent is prediction_market
            if intent.intent_type == "prediction_market":
                from finclaw.agent.financial.prompts import PREDICTION_ROUTING_PROMPT
                finance_context_parts.append(PREDICTION_ROUTING_PROMPT)

            # Check financial history first
            history_hits = self.fin_history.search(
                tickers=intent.tickers,
                intent_type=intent.intent_type,
            )
            if history_hits:
                history_lines = "\n".join(
                    [f"- {hit['title']}\n  {hit['content']}" for hit in history_hits]
                )
                finance_context_parts.append(
                    "[System Context - Relevant Financial History]\n\n"
                    f"{history_lines}"
                )

            # Then check reusable cache entries
            cache_hits = self.fin_cache.lookup(
                tickers=intent.tickers,
                task_type=intent.intent_type,
            )
            if cache_hits:
                from finclaw.agent.financial.prompts import CACHE_REUSE_CONTEXT_PROMPT
                summaries = []
                for hit in cache_hits:
                    summaries.append(
                        f"- [{hit['created_at']}] {hit['summary']}\n"
                        f"  Analysis file: {hit.get('analysis_file', 'N/A')}"
                    )
                finance_context_parts.append(
                    CACHE_REUSE_CONTEXT_PROMPT.format(cached_results="\n".join(summaries))
                )

        self._set_tool_context(msg.channel, msg.chat_id)

        # Prepend finance context to user message if available
        user_content = msg.content
        if finance_context_parts:
            finance_context = "\n\n".join(finance_context_parts)
            user_content = f"{finance_context}\n\n{user_content}"

        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=user_content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        final_content, tools_used = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        session.add_message("user", original_user_content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        # Financial post-hook: record to financial history and cache index
        if intent.is_financial and final_content:
            try:
                history_metadata: dict[str, Any] = {}

                # Persist analysis output for token/time reuse on repeated questions.
                if intent.tickers:
                    primary_ticker = intent.tickers[0]
                    analysis_file = None
                    if intent.intent_type != "price_query":
                        analysis_file = self.fin_cache.save_analysis(
                            ticker=primary_ticker,
                            topic=intent.intent_type,
                            analysis={
                                "query": original_user_content,
                                "response": final_content,
                                "intent_type": intent.intent_type,
                                "tickers": intent.tickers,
                                "tools_used": tools_used or [],
                            },
                        )
                    index_entry = self.fin_cache.add_index_entry(
                        ticker=primary_ticker,
                        task_type=intent.intent_type,
                        query=original_user_content,
                        summary=final_content,
                        analysis_file=analysis_file,
                    )
                    history_metadata["analysis_file"] = analysis_file
                    if index_entry:
                        history_metadata["index_id"] = index_entry.get("id")
                        history_metadata["history_ref"] = index_entry.get("history_ref")
                        if index_entry.get("period"):
                            history_metadata["period"] = index_entry["period"]
                        if index_entry.get("raw_files"):
                            history_metadata["raw_files"] = index_entry["raw_files"]
                elif intent.intent_type == "prediction_market":
                    index_entry = self.fin_cache.add_index_entry(
                        ticker="PREDICTION_MARKET",
                        task_type=intent.intent_type,
                        query=original_user_content,
                        summary=final_content,
                    )
                    if index_entry:
                        history_metadata["index_id"] = index_entry.get("id")
                        history_metadata["history_ref"] = index_entry.get("history_ref")

                await self.fin_history.add_entry(
                    query=original_user_content,
                    response=final_content,
                    intent=intent,
                    tools_used=tools_used,
                    metadata=history_metadata,
                )
            except Exception as e:
                logger.error(f"Financial history write failed: {e}")

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _ = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            if entry := result.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

        # Financial history consolidation (triggered when content exceeds 10KB)
        try:
            fin_history_content = self.fin_history.read_all()
            if fin_history_content and len(fin_history_content) > 10000:
                from finclaw.agent.financial.prompts import FINANCIAL_CONSOLIDATION_PROMPT
                compress_response = await self.provider.chat(
                    messages=[
                        {"role": "system", "content": "You are a financial data archival expert. Respond only with the compressed content."},
                        {"role": "user", "content": FINANCIAL_CONSOLIDATION_PROMPT.format(
                            financial_history=fin_history_content
                        )},
                    ],
                    model=self.model,
                )
                if compress_response.content:
                    self.fin_history.write_all(compress_response.content.strip())
                    logger.info("Financial history consolidation done")
        except Exception as e:
            logger.error(f"Financial history consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
        
        Returns:
            The agent's response.
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""
