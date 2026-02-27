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

    @staticmethod
    def _extract_data_excerpt(payload: Any, max_items: int = 5) -> str:
        """Extract a compact JSON excerpt from raw tool output.

        Walks common response shapes (list of dicts, dict with 'markets' key,
        list of such dicts) and returns a truncated JSON string the main agent
        can use directly — no cache file reading required.
        """
        def _pick_items(obj: Any) -> list[dict] | None:
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return obj
            if isinstance(obj, dict):
                for key in ("markets", "results", "data", "events", "matches"):
                    items = obj.get(key)
                    if isinstance(items, list) and items:
                        return items
            return None

        # payload may be a single result or a list of results from multiple tools
        sources = payload if isinstance(payload, list) else [payload]
        excerpts: list[Any] = []
        for src in sources:
            items = _pick_items(src)
            if items:
                excerpts.append(items[:max_items])
            elif isinstance(src, dict):
                # Grab summary/detail keys if present
                for key in ("summary", "detail", "comparison"):
                    if key in src:
                        excerpts.append(src[key])

        if not excerpts:
            return ""

        compact = excerpts[0] if len(excerpts) == 1 else excerpts
        return json.dumps(compact, ensure_ascii=False, default=str)

    async def _run_inner_agent(self, query: str) -> str:
        """
        Agentic inner loop: the inner LLM can chain multiple tool calls across
        turns (e.g. search → market_history) before producing its synthesis.

        Loop (up to _MAX_INNER_TURNS):
          • Each turn: call LLM with tools available.
          • If no tool calls → the LLM's content is the final synthesis; stop.
          • If tool calls → execute concurrently, append results, continue.
        After the loop (max turns hit without natural stop) → force one synthesis
        call without tools.
        """
        inner_tools = self._build_inner_tools()
        tool_map: dict[str, Tool] = {t.name: t for t in inner_tools}
        tool_defs: list[dict[str, Any]] = [t.to_schema() for t in inner_tools]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._inner_system_prompt},
            {"role": "user", "content": query},
        ]

        _MAX_INNER_TURNS = 4
        _MAX_TOOL_RESULT_CHARS = 12_000

        all_raw_items: list[Any] = []
        all_tool_errors: list[str] = []
        all_tool_names: list[str] = []
        summary = ""

        async def _exec(tc_name: str, tc_args: dict[str, Any]) -> str:
            if tc_name not in tool_map:
                return json.dumps({"error": f"unknown inner tool '{tc_name}'"})
            try:
                return await tool_map[tc_name].execute(**tc_args)
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": str(exc)})

        for _turn in range(_MAX_INNER_TURNS):
            response = await self._inner_provider.chat(
                messages=messages,
                tools=tool_defs,
                model=self._model,
                temperature=0.1,
                max_tokens=1024,
            )

            if not response.has_tool_calls:
                # Inner LLM is done — its content is the synthesis.
                summary = response.content or ""
                break

            # ── Append assistant tool-call message ───────────────────────────
            tool_call_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": tool_call_dicts,
            })

            # ── Execute all tool calls concurrently ──────────────────────────
            results: list[str] = await asyncio.gather(
                *[_exec(tc.name, tc.arguments) for tc in response.tool_calls]
            )

            # ── Append tool results (capped) to keep context bounded ─────────
            for tc, result in zip(response.tool_calls, results):
                content = result
                if len(content) > _MAX_TOOL_RESULT_CHARS:
                    content = content[:_MAX_TOOL_RESULT_CHARS] + "\n…[truncated]"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": content,
                })

            # ── Collect raw data + errors across turns ───────────────────────
            for tc, result in zip(response.tool_calls, results):
                all_tool_names.append(tc.name)
                try:
                    parsed = json.loads(result)
                    all_raw_items.append(parsed)
                    if isinstance(parsed, dict):
                        if "error" in parsed:
                            all_tool_errors.append(f"{tc.name}: {parsed['error']}")
                        elif parsed.get("success") is False:
                            all_tool_errors.append(
                                f"{tc.name}: {parsed.get('error', 'failed (no details)')}"
                            )
                except Exception:
                    all_raw_items.append(result)
        else:
            # Max turns exhausted without natural synthesis → force one final call.
            _SYNTHESIS_SKIP_THRESHOLD = 50_000  # bytes
            forced_raw = json.dumps(
                all_raw_items[0] if len(all_raw_items) == 1 else all_raw_items,
                ensure_ascii=False,
                default=str,
            )
            if len(forced_raw.encode()) <= _SYNTHESIS_SKIP_THRESHOLD:
                if all_tool_errors:
                    messages.append({
                        "role": "user",
                        "content": (
                            "TOOL ERRORS (quote these exactly in your response):\n"
                            + "\n".join(all_tool_errors)
                        ),
                    })
                synthesis = await self._inner_provider.chat(
                    messages=messages,
                    model=self._model,
                    temperature=0.2,
                    max_tokens=800,
                )
                summary = synthesis.content or ""

        payload = all_raw_items[0] if len(all_raw_items) == 1 else all_raw_items

        logger.debug(
            f"{self.name} inner-agent: turns={_turn + 1} calls={all_tool_names}"
            + (f" errors={all_tool_errors}" if all_tool_errors else "")
        )

        # ── Serialise raw payload for cache ──────────────────────────────────
        raw_json = json.dumps(payload, ensure_ascii=False, default=str)

        # ── Fallback: if synthesis is empty, extract key data inline ─────────
        # The main agent should never receive an empty answer — always surface
        # a compact data excerpt so it can draw conclusions without reading cache.
        if not summary.strip():
            excerpt = self._extract_data_excerpt(payload)
            if excerpt:
                summary = excerpt

        # ── Save raw data to workspace cache ─────────────────────────────────
        data_file: str | None = None
        if self._workspace is not None:
            try:
                cache_dir = self._workspace / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fpath = cache_dir / f"{self.name}_{ts}.json"
                await asyncio.to_thread(fpath.write_text, raw_json, encoding="utf-8")
                data_file = str(fpath)
            except Exception as exc:
                logger.warning(f"{self.name}: failed to cache raw data: {exc}")

        # ── Return plain text so the outer agent can quote it directly ────────
        # Returning a JSON wrapper with {"summary":..., "data_file":...} causes
        # the outer agent to always infer "synthesis failed / data spilled to
        # file" because data_file is present.  Plain text avoids that confusion.
        if summary.strip():
            output = summary
            if all_tool_errors:
                output += " | Errors: " + "; ".join(all_tool_errors)
            if data_file:
                output += f"\n\n(Full raw data: {data_file})"
            return output

        # Nothing usable — return a minimal error JSON so the outer agent knows
        if all_tool_errors:
            return json.dumps({"error": all_tool_errors, "data_file": data_file}, default=str)
        return json.dumps({"error": "no data returned", "data_file": data_file}, default=str)
