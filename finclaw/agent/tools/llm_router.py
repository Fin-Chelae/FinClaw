"""LLM Router Tool — Dexter-pattern inner LLM sub-agent base class.

Instead of exposing raw parameter dispatch to the main LLM, each router accepts
a natural-language query and internally runs a focused inner LLM sub-agent that
has access to specialized data tools, calls them (in parallel where possible),
and returns the combined result.

Usage:
    class MyRouter(LLMRouterTool):
        name = "my_router"
        description = "..."
        parameters = {"type": "object", "properties": {"query": {...}}, "required": ["query"]}
        _inner_system_prompt = "You are a specialist..."

        def _build_inner_tools(self) -> list[Tool]:
            return [DataToolA(), DataToolB()]

        async def execute(self, **kwargs) -> str:
            return await self._run_inner_agent(kwargs.get("query", ""))
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from finclaw.agent.tools.base import Tool
from finclaw.providers.base import LLMProvider


class LLMRouterTool(Tool):
    """
    Base class for router tools backed by an inner LLM sub-agent (Dexter pattern).

    The main LLM calls this tool with a natural-language query. Internally:
      1. One inner LLM call selects which data tools to call (may select multiple).
      2. All selected tools execute concurrently (asyncio.gather).
      3. One synthesis call produces a concise summary; raw data is saved to a
         workspace cache file to keep the main agent context lean.

    Subclasses must define:
      - name, description, parameters  — tool schema exposed to the main LLM
      - _inner_system_prompt           — instructions for the inner LLM
      - _build_inner_tools()           — data tools available to the inner LLM
    """

    _inner_system_prompt: str = "You are a data retrieval specialist."

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        inner_model: str = "",
        inner_provider: LLMProvider | None = None,
        workspace: Path | None = None,
    ) -> None:
        self._provider = provider
        self._inner_provider = inner_provider or provider
        self._model = inner_model or model
        self._workspace = workspace

    def _build_inner_tools(self) -> list[Tool]:
        """Return data tools available to the inner LLM. Override in subclasses."""
        return []

    async def _run_inner_agent(self, query: str) -> str:
        """
        Route query to data tools in a single LLM call, then produce a
        structured response containing both a concise summary and the raw data:

          1. One call to select tools (may pick multiple for parallel execution).
          2. Execute all selected tools concurrently.
          3. One synthesis call → short summary (the main agent uses this for
             analysis; the raw data is preserved for reflection / follow-up).
        """
        inner_tools = self._build_inner_tools()
        tool_map: dict[str, Tool] = {t.name: t for t in inner_tools}
        tool_defs: list[dict[str, Any]] = [t.to_schema() for t in inner_tools]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._inner_system_prompt},
            {"role": "user", "content": query},
        ]

        # ── Step 1: tool selection ────────────────────────────────────────────
        selection = await self._inner_provider.chat(
            messages=messages,
            tools=tool_defs,
            model=self._model,
            temperature=0.1,
            max_tokens=1024,
        )

        if not selection.has_tool_calls:
            return selection.content or json.dumps({"status": "no_result"})

        # ── Step 2: execute all tool calls concurrently ───────────────────────
        tool_call_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in selection.tool_calls
        ]
        messages.append({
            "role": "assistant",
            "content": selection.content or "",
            "tool_calls": tool_call_dicts,
        })

        async def _exec(tc_name: str, tc_args: dict[str, Any]) -> str:
            if tc_name not in tool_map:
                return json.dumps({"error": f"unknown inner tool '{tc_name}'"})
            try:
                return await tool_map[tc_name].execute(**tc_args)
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": str(exc)})

        results: list[str] = await asyncio.gather(
            *[_exec(tc.name, tc.arguments) for tc in selection.tool_calls]
        )

        for tc, result in zip(selection.tool_calls, results):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": result,
            })

        # ── Collect raw data + detect errors ─────────────────────────────────
        raw_items: list[Any] = []
        tool_errors: list[str] = []
        for tc, result in zip(selection.tool_calls, results):
            try:
                parsed = json.loads(result)
                raw_items.append(parsed)
                if isinstance(parsed, dict):
                    if "error" in parsed:
                        tool_errors.append(f"{tc.name}: {parsed['error']}")
                    elif parsed.get("success") is False:
                        tool_errors.append(f"{tc.name}: {parsed.get('error', 'failed (no details)')}")
            except Exception:
                raw_items.append(result)
        payload = raw_items[0] if len(raw_items) == 1 else raw_items

        logger.debug(
            f"{self.name} inner-agent: calls={[tc.name for tc in selection.tool_calls]}"
            + (f" errors={tool_errors}" if tool_errors else "")
        )

        # ── Serialise raw payload (needed for both cache and size check) ─────
        raw_json = json.dumps(payload, ensure_ascii=False, default=str)
        raw_size = len(raw_json.encode())

        # ── Step 3: concise summary (skipped when payload is very large) ─────
        # When the tool result is too large for the synthesis LLM to process
        # within the max_tokens budget the outer agent gets the data_file path
        # and can use read_file to inspect the raw data directly.
        _SYNTHESIS_SKIP_THRESHOLD = 50_000  # bytes
        skip_synthesis = raw_size > _SYNTHESIS_SKIP_THRESHOLD

        if skip_synthesis:
            logger.info(
                f"{self.name}: payload {raw_size:,} bytes exceeds synthesis threshold "
                f"({_SYNTHESIS_SKIP_THRESHOLD:,}), skipping inline synthesis"
            )
            summary = (
                f"Data payload is {raw_size:,} bytes — too large for inline synthesis. "
                "The raw data has been saved to data_file. "
                "Use read_file on data_file to analyse outcomes, probabilities, and history directly."
            )
        else:
            # If errors were detected, inject them explicitly so the synthesis LLM
            # quotes them verbatim rather than paraphrasing.
            if tool_errors:
                error_block = "TOOL ERRORS (quote these exactly in your response):\n" + "\n".join(tool_errors)
                messages.append({"role": "user", "content": error_block})

            synthesis = await self._inner_provider.chat(
                messages=messages,
                model=self._model,
                temperature=0.2,
                max_tokens=400,
            )
            summary = synthesis.content or ""

        # ── Save raw data to workspace cache ─────────────────────────────────
        # Kept out of main-agent context until explicitly needed via read_file.
        out: dict[str, Any] = {"summary": summary}
        if tool_errors:
            out["errors"] = tool_errors
        if self._workspace is not None:
            try:
                cache_dir = self._workspace / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fpath = cache_dir / f"{self.name}_{ts}.json"
                await asyncio.to_thread(fpath.write_text, raw_json, encoding="utf-8")
                out["data_file"] = str(fpath)
            except Exception as exc:
                logger.warning(f"{self.name}: failed to cache raw data: {exc}")
                out["data"] = payload
        else:
            out["data"] = payload

        return json.dumps(out, ensure_ascii=False, default=str)
