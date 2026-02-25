"""LiteLLM provider implementation for multi-provider support."""

import asyncio
import json
import json_repair
import logging
import os
from datetime import datetime
from typing import Any

import litellm
from litellm import acompletion
from loguru import logger

from finclaw.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from finclaw.providers.registry import find_by_model, find_gateway


def _is_verbose() -> bool:
    """Check whether verbose LLM logging is enabled in config."""
    try:
        from finclaw.config.loader import load_config
        return load_config().agents.defaults.log_verbose
    except Exception:
        return False


def _fmt_messages(messages: list[dict[str, Any]]) -> str:
    """Pretty-format messages list for human-readable logging."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content")
        # Content can be a string or a list of content blocks
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        text_parts.append("[image]")
                    else:
                        text_parts.append(f"[{block.get('type', '?')}]")
                else:
                    text_parts.append(str(block))
            content = "\n".join(text_parts)
        elif content is None:
            content = ""
        parts.append(f"  [{role}]\n{_indent(content, 4)}")
    return "\n".join(parts)


def _fmt_tool_calls(tool_calls: list[ToolCallRequest]) -> str:
    """Pretty-format tool calls for logging."""
    parts: list[str] = []
    for tc in tool_calls:
        args_str = json.dumps(tc.arguments, ensure_ascii=False, indent=2)
        parts.append(f"  {tc.name}({tc.id}):\n{_indent(args_str, 4)}")
    return "\n".join(parts)


def _fmt_tools(tools: list[dict[str, Any]]) -> str:
    """Compact summary of available tools."""
    names = [t.get("function", {}).get("name", "?") for t in tools]
    return ", ".join(names)


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        
        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, api_key, api_base)
        
        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model)
        
        if api_base:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True
    
    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return
        if not spec.env_key:
            # OAuth/provider-only specs (for example: openai_codex)
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model
        
        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"
        
        return model
    
    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        model = self._resolve_model(model or self.default_model)
        
        # Clamp max_tokens to at least 1 — negative or zero values cause
        # LiteLLM to reject the request with "max_tokens must be at least 1".
        max_tokens = max(1, max_tokens)
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)
        
        # Pass api_key directly — more reliable than env vars alone
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        # Pass api_base for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        verbose = _is_verbose()

        if verbose:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            tools_line = f"║  Tools: [{_fmt_tools(tools)}]\n" if tools else ""
            sep = "═" * 60
            logger.info(
                f"\n╔══ LLM REQUEST ══ {model} ══ {ts} ══\n"
                f"║ Messages:\n{_fmt_messages(messages)}\n"
                f"{tools_line}"
                f"╚{sep}"
            )

        _max_retries = 3
        for _attempt in range(_max_retries + 1):
            try:
                response = await acompletion(**kwargs)
                parsed = self._parse_response(response)
                self._log_token_usage(model, messages, parsed)

                if verbose:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    usage = parsed.usage or {}
                    sep = "═" * 60
                    resp_parts = [
                        f"\n╔══ LLM RESPONSE ══ {model} ══ {ts} ══",
                        f"║ Finish: {parsed.finish_reason}  "
                        f"Tokens: {usage.get('prompt_tokens', '?')} in / "
                        f"{usage.get('completion_tokens', '?')} out",
                    ]
                    if parsed.reasoning_content:
                        resp_parts.append(f"║ Reasoning:\n{_indent(parsed.reasoning_content, 4)}")
                    if parsed.content:
                        resp_parts.append(f"║ Content:\n{_indent(parsed.content, 4)}")
                    if parsed.tool_calls:
                        resp_parts.append(f"║ Tool Calls:\n{_fmt_tool_calls(parsed.tool_calls)}")
                    resp_parts.append(f"╚{sep}")
                    logger.info("\n".join(resp_parts))

                return parsed
            except litellm.RateLimitError as e:
                if _attempt == _max_retries:
                    return LLMResponse(
                        content=f"Error calling LLM: {str(e)}",
                        finish_reason="error",
                    )
                wait = 5 * (2 ** _attempt)  # 5s → 10s → 20s
                logger.warning(
                    f"Rate limit hit (attempt {_attempt + 1}/{_max_retries}), "
                    f"retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return LLMResponse(
                    content=f"Error calling LLM: {str(e)}",
                    finish_reason="error",
                )
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json_repair.loads(args)
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        reasoning_content = getattr(message, "reasoning_content", None)
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model

    def _log_token_usage(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response: LLMResponse,
    ) -> None:
        """Log per-request token usage for system diagnostics."""
        usage = response.usage or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        source = "api"

        if input_tokens is None:
            input_tokens = self._estimate_message_tokens(messages, model)
            source = "estimated"
        if output_tokens is None:
            output_tokens = self._estimate_text_tokens(response.content or "", model)
            source = "estimated"

        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens

        logger.info(
            "llm_tokens provider=litellm model={} input_tokens={} output_tokens={} total_tokens={} source={}",
            model,
            input_tokens,
            output_tokens,
            total_tokens,
            source,
        )

    def _estimate_message_tokens(self, messages: list[dict[str, Any]], model: str) -> int:
        try:
            count = litellm.token_counter(model=model, messages=messages)
            return int(count) if count is not None else 0
        except Exception:
            raw = json.dumps(messages, ensure_ascii=False)
            return max(1, len(raw) // 4)

    def _estimate_text_tokens(self, text: str, model: str) -> int:
        if not text:
            return 0
        try:
            count = litellm.token_counter(model=model, text=text)
            return int(count) if count is not None else 0
        except Exception:
            return max(1, len(text) // 4)
