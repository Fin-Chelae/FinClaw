"""Shell execution tool."""

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from finclaw.agent.tools.base import Tool


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        max_output_len: int = 10000,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"\b(format|mkfs|diskpart)\b",   # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
            # Interpreter invocations (arbitrary code execution)
            r"\b(python[0-9]*|perl|ruby|node)\b",
            # Subshell evasion
            r"\b(bash|sh|zsh)\s+-c\b",
            # find with dangerous actions
            r"\bfind\b.*-(delete|exec)",
            # Remote code execution via pipe to shell
            r"(curl|wget).*\|.*(sh|bash)",
            # Encoded payload execution
            r"base64.*\|.*(sh|bash)",
            # xargs with rm
            r"\bxargs\b.*\brm\b",
            # World-writable permissions
            r"\bchmod\b.*777",
            # Privilege escalation
            r"\bsudo\b",
            r"\bsu\s+-",
            # Reverse shell tools
            r"\b(nc|ncat|netcat|socat)\b",
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.max_output_len = max_output_len

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command"
                }
            },
            "required": ["command"]
        }

    def _sanitize_env(self) -> dict[str, str]:
        """Return a copy of os.environ with sensitive keys removed."""
        sensitive_patterns = [
            r".*_KEY$",
            r".*_TOKEN$",
            r".*_SECRET$",
            r".*_PASSWORD$",
            r".*PRIVATE.*",
            r".*CREDENTIAL.*",
        ]
        sanitized = {}
        for key, value in os.environ.items():
            if not any(re.match(pattern, key, re.IGNORECASE) for pattern in sensitive_patterns):
                sanitized[key] = value
        return sanitized

    async def execute(self, command: str, working_dir: str | None = None, **kwargs: Any) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=self._sanitize_env(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return f"Error: Command timed out after {self.timeout} seconds"

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            if process.returncode != 0:
                output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Truncate very long output
            if len(result) > self.max_output_len:
                result = result[:self.max_output_len] + f"\n... (truncated, {len(result) - self.max_output_len} more chars)"

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            # Check for symlink creation (potential escape vector)
            if re.search(r"\bln\s+-s\b", lower):
                return "Error: Command blocked by safety guard (symlink creation not allowed in workspace mode)"

            cwd_path = Path(cwd).resolve()

            win_paths = re.findall(r"[A-Za-z]:\\[^\\\"']+", cmd)
            # Only match absolute paths — avoid false positives on relative
            # paths like ".venv/bin/python" where "/bin/python" would be
            # incorrectly extracted by the old pattern.
            posix_paths = re.findall(r"(?:^|[\s|>])(/[^\s\"'>]+)", cmd)

            for raw in win_paths + posix_paths:
                try:
                    p = Path(raw.strip()).resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None
