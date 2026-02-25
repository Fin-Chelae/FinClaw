"""CLI commands for finclaw."""

import asyncio
import os
import signal
from pathlib import Path
import select
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from finclaw import __version__, __logo__
from finclaw.config.schema import Config

app = typer.Typer(
    name="finclaw",
    help=f"{__logo__} finclaw - Financial Analysis Expert",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input: prompt_toolkit for editing, paste, history, and display
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    history_file = Path.home() / ".finclaw" / "history" / "cli_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,   # Enter submits (single line mode)
    )


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """Render assistant response with consistent terminal styling."""
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(f"[cyan]{__logo__} FinClaw[/cyan]")
    console.print(body)
    console.print()


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit (handles paste, history, display).

    prompt_toolkit natively handles:
    - Multiline paste (bracketed paste mode)
    - History navigation (up/down arrows)
    - Clean display (no ghost characters or artifacts)
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc



def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} finclaw v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """finclaw - Financial Analysis Expert."""
    pass


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard():
    """Initialize finclaw configuration and workspace."""
    from finclaw.config.loader import get_config_path, load_config, save_config
    from finclaw.config.schema import Config
    from finclaw.utils.helpers import get_workspace_path
    
    config_path = get_config_path()

    # --- Config ---
    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("  [bold]y[/bold] = overwrite with defaults  [bold]N[/bold] = refresh (keep existing values)")
        if typer.confirm("Overwrite config?", default=False):
            save_config(Config())
            console.print(f"[green]✓[/green] Config reset to defaults")
        else:
            config = load_config()
            save_config(config)
            console.print(f"[green]✓[/green] Config refreshed (existing values preserved)")
    else:
        save_config(Config())
        console.print(f"[green]✓[/green] Created config at {config_path}")

    # --- Workspace ---
    workspace = get_workspace_path()

    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace}")

    # --- Workspace files (SOUL.md, memory, financial_data, etc.) ---
    force_workspace = False
    if (workspace / "SOUL.md").exists():
        console.print(f"\n[yellow]Workspace files already exist at {workspace}[/yellow]")
        console.print("  [bold]y[/bold] = reset all to defaults (SOUL.md, memory, data cache, etc.)")
        console.print("  [bold]N[/bold] = keep existing, only create missing files")
        force_workspace = typer.confirm("Reset workspace files?", default=False)
        if force_workspace:
            console.print(f"[green]✓[/green] Resetting workspace files to defaults")

    _create_workspace_templates(workspace, force=force_workspace)
    
    console.print(f"\n{__logo__} finclaw is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your API key to [cyan]~/.finclaw/config.json[/cyan]")
    console.print("     Get one at: https://openrouter.ai/keys")
    console.print("  2. Chat: [cyan]finclaw agent -m \"Hello!\"[/cyan]")
    console.print("\n[dim]Optional — Meme monitor setup:[/dim]")
    console.print("  [dim]Twitter cookies: set [cyan]tools.memeMonitor.twitterCookies[/cyan] in config.json[/dim]")
    console.print("  [dim]Self-hosted RSSHub: set [cyan]tools.memeMonitor.rsshubBaseUrl[/cyan] in config.json[/dim]")
    console.print("\n[dim]Want Telegram/WhatsApp? See configuration docs.[/dim]")




def _create_workspace_templates(workspace: Path, force: bool = False):
    """Create default workspace template files. When force=True, overwrite all existing files and reset data."""
    templates = {
        "AGENTS.md": """# Agent Instructions

You are FinClaw, a helpful financial analysis expert. Be concise, data-driven, and accurate.

## Core Capabilities

- Stock/crypto price queries and market data analysis
- Financial statement analysis (income statement, balance sheet, cash flow)
- Company valuation and fundamental analysis
- Macroeconomic analysis: GDP, CPI, unemployment, Fed policy, yield curve, FX, commodities
- Market trend, sector analysis, and news search
- Read, write, and edit files; execute shell commands; search the web

## Guidelines

- Lead with conclusions, followed by key data and uncertainty notes
- For price/earnings data, rely on tool results or cached data — never fabricate numbers
- Always label data timestamps and reporting periods (e.g., "Q3 2025, reported 2025-11-20")
- When cached analysis results are available, reference them to avoid redundant token usage
- Retain general-purpose assistant capabilities — not restricted to finance only
- Remember important information in memory/MEMORY.md
- Financial analysis history goes to memory/FINANCIAL_HISTORY.md

## Response Formatting

- Present data in clean, user-friendly language — never expose raw tool output, internal field names, timestamps, JSON keys, or API details
- Convert all timestamps to readable dates/times using the Current Time section for timezone context
- Do not mention tool names, data source names, or routing logic
- Keep responses concise for simple queries
""",
        "TOOLS.md": """# Available Tools

This document describes the tools available to FinClaw.

## Tool Routing Guide

- Stock prices, financials, ratios → `financial_metrics`
- Equity valuation (DCF, DDM, multiples) → `equity_valuation`
- Macro data fetch (FRED series, FX rates, yields, commodities, release calendar) → `economics_data`
- Macro analysis models (business cycle, policy, growth, trade, FX analysis, ARIMA) → `economics_analysis`
- Company/ticker search → `financial_search`
- Schedule reminders and recurring tasks → `cron`
- General web lookup → `web_search` (use ONLY when dedicated tools don't cover the query)

## File Operations

- `read_file(path)` — Read file contents
- `write_file(path, content)` — Write content to a file
- `edit_file(path, old_text, new_text)` — Edit a file by replacing text
- `list_dir(path)` — List directory contents

## Shell Execution

- `exec(command, working_dir=None)` — Execute a shell command (timeout 60s, dangerous commands blocked)

## Web Access

- `web_search(query, count=5)` — Search the web
- `web_fetch(url, extractMode="markdown", maxChars=50000)` — Fetch and extract content from a URL

## Financial Tools

- `financial_metrics(ticker, metrics, period="annual", limit=5)` — Query company financial data
- `financial_search(query, search_type, ticker=None, date_range=None)` — Search news, filings, company facts
- `equity_valuation(ticker, method, params)` — Run valuation models (DCF, DDM, multiples)
- `economics_data(command, ...)` — Fetch macro data: FRED series, FX rates, yields, commodities, economic calendar
- `economics_analysis(model, ...)` — Run CFA-level analysis: cycles, policy, growth, trade, FX, ARIMA, Monte Carlo

## Scheduling

- `cron(action, ...)` — Schedule one-time or recurring tasks. Actions: add, list, remove. Use cron_expr for daily/weekly, at for one-time, every_seconds for intervals.

## Communication

- `message(content, channel=None, chat_id=None)` — Send a message to a chat channel

## Background Tasks

- `spawn(task, label=None)` — Spawn a subagent for background work
""",
        "SOUL.md": """# Soul

I am FinClaw, a helpful financial analysis expert.

## Personality

- Professional and data-driven
- Concise and to the point: conclusions first, then supporting data
- Cautious: always note data staleness and uncertainty

## Values

- Accuracy over speed
- Be genuinely helpful. Actions speak louder than filler words.
- Be resourceful before asking: The goal is to come back with answers, not questions.
- Be thorough and efficient
- Data-backed analysis over speculation
- User privacy and financial data security

## Communication Style

- Be clear and direct. Do not over-clarify or repeatedly ask for confirmation when the intent is reasonably clear.
- Lead with key conclusions and numbers
- Flag uncertainty and data freshness

## Response Format

- Keep casual responses brief and direct
- For research: lead with the key finding and include specific data points
- For non-comparative information, prefer plain text or simple lists over tables
- Don't narrate your actions or ask leading questions about what the user wants

""",
        "USER.md": """# User

Information about the user goes here.

## Preferences

- Communication style: (casual/formal)
- Timezone: (your timezone)
- Language: (your preferred language)
""",
    }
    
    for filename, content in templates.items():
        file_path = workspace / filename
        if force or not file_path.exists():
            file_path.write_text(content)
            console.print(f"  [dim]{'Reset' if force else 'Created'} {filename}[/dim]")
    
    # Create memory directory and MEMORY.md
    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)
    memory_file = memory_dir / "MEMORY.md"
    if force or not memory_file.exists():
        memory_file.write_text("""# Long-term Memory

This file stores important information that should persist across sessions.

## User Information

(Important facts about the user)

## Preferences

(User preferences learned over time)

## Important Notes

(Things to remember)
""")
        console.print(f"  [dim]{'Reset' if force else 'Created'} memory/MEMORY.md[/dim]")
    
    history_file = memory_dir / "HISTORY.md"
    if force or not history_file.exists():
        history_file.write_text("")
        console.print(f"  [dim]{'Reset' if force else 'Created'} memory/HISTORY.md[/dim]")

    # Create skills directory for custom user skills
    skills_dir = workspace / "skills"
    skills_dir.mkdir(exist_ok=True)

    # Financial history file
    fin_history = memory_dir / "FINANCIAL_HISTORY.md"
    if force or not fin_history.exists():
        fin_history.write_text("# Financial Analysis History\n\n")
        console.print(f"  [dim]{'Reset' if force else 'Created'} memory/FINANCIAL_HISTORY.md[/dim]")

    # Financial data cache directory
    fin_data_dir = workspace / "financial_data"
    if force and fin_data_dir.exists():
        import shutil
        shutil.rmtree(fin_data_dir)
        console.print("  [dim]Reset financial_data/ cache[/dim]")
    fin_data_dir.mkdir(exist_ok=True)
    (fin_data_dir / "raw").mkdir(exist_ok=True)
    (fin_data_dir / "analysis").mkdir(exist_ok=True)

    index_path = fin_data_dir / "index.json"
    if force or not index_path.exists():
        index_path.write_text('{"entries": []}')
        if not force:
            console.print("  [dim]Created financial_data/index.json[/dim]")


def _make_provider(config: Config):
    """Create LiteLLMProvider from config. Exits if no API key found."""
    from finclaw.providers.litellm_provider import LiteLLMProvider
    from finclaw.providers.openai_codex_provider import OpenAICodexProvider

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    # OpenAI Codex (OAuth): don't route via LiteLLM; use the dedicated implementation.
    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return OpenAICodexProvider(default_model=model)

    if not model.startswith("bedrock/") and not (p and p.api_key):
        console.print("[red]Error: No API key configured.[/red]")
        console.print("Set one in ~/.finclaw/config.json under providers section")
        raise typer.Exit(1)

    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=provider_name,
    )


def _make_inner_provider(config: Config):
    """Create a separate LiteLLMProvider for inner LLM sub-agents.

    Returns None if inner_model is empty or uses the same provider as the main model.
    """
    from finclaw.providers.litellm_provider import LiteLLMProvider

    inner_model = config.agents.defaults.inner_model
    if not inner_model:
        return None

    inner_provider_name = config.get_provider_name(inner_model)
    main_provider_name = config.get_provider_name(config.agents.defaults.model)
    if inner_provider_name == main_provider_name:
        return None  # Same provider — main provider handles it fine

    p = config.get_provider(inner_model)
    if not (p and p.api_key):
        console.print(
            f"[yellow]⚠ inner_model '{inner_model}' configured but no API key found "
            f"for provider '{inner_provider_name}'. Falling back to main provider.[/yellow]"
        )
        return None

    return LiteLLMProvider(
        api_key=p.api_key,
        api_base=config.get_api_base(inner_model),
        default_model=inner_model,
        extra_headers=p.extra_headers if p else None,
        provider_name=inner_provider_name,
    )


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command()
def gateway(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the finclaw gateway."""
    from finclaw.config.loader import load_config, get_data_dir
    from finclaw.bus.queue import MessageBus
    from finclaw.agent.loop import AgentLoop
    from finclaw.channels.manager import ChannelManager
    from finclaw.session.manager import SessionManager
    from finclaw.cron.service import CronService
    from finclaw.cron.types import CronJob
    from finclaw.heartbeat.service import HeartbeatService
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        # Silence noisy HTTP transport logs
        for noisy in ("httpcore", "httpx", "hpack"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    
    console.print(f"{__logo__} Starting finclaw gateway on port {port}...")
    
    config = load_config()

    # Debug: print API key info for troubleshooting
    import os as _os
    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    config_key = p.api_key if p else ""
    env_key = _os.environ.get("ANTHROPIC_API_KEY", "")
    console.print(f"[dim]Model: {model}[/dim]")
    console.print(f"[dim]Provider: {provider_name or 'unknown'}[/dim]")
    console.print(f"[dim]Config API key: {'...' + config_key[-8:] if len(config_key) > 8 else '(empty)'}[/dim]")
    console.print(f"[dim]Env ANTHROPIC_API_KEY: {'...' + env_key[-8:] if len(env_key) > 8 else '(not set)'}[/dim]")
    if env_key and config_key and env_key != config_key:
        console.print("[yellow]⚠ Env var differs from config — env var takes precedence![/yellow]")

    bus = MessageBus()
    provider = _make_provider(config)
    inner_provider = _make_inner_provider(config)
    session_manager = SessionManager(config.workspace_path)

    # Create cron service first (callback set after agent creation)
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # Create agent with cron service
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        inner_model=config.agents.defaults.inner_model,
        inner_provider=inner_provider,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        search_api_key=config.tools.web.search.api_key or None,
        search_provider=config.tools.web.search.provider,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
    )
    
    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent."""
        response = await agent.process_direct(
            job.payload.message,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
        )
        if job.payload.deliver and job.payload.to:
            from finclaw.bus.events import OutboundMessage
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to,
                content=response or ""
            ))
        return response
    cron.on_job = on_cron_job
    
    # Create heartbeat service
    async def on_heartbeat(prompt: str) -> str:
        """Execute heartbeat through the agent."""
        return await agent.process_direct(prompt, session_key="heartbeat")
    
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        on_heartbeat=on_heartbeat,
        interval_s=30 * 60,  # 30 minutes
        enabled=True
    )
    
    # Create channel manager
    channels = ChannelManager(config, bus)
    
    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")
    
    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")
    
    console.print(f"[green]✓[/green] Heartbeat: every 30m")
    
    async def run():
        try:
            await cron.start()
            await heartbeat.start()
            await asyncio.gather(
                agent.run(),
                channels.start_all(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            await agent.close_mcp()
            heartbeat.stop()
            cron.stop()
            agent.stop()
            await channels.stop_all()
    
    asyncio.run(run())




# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show finclaw runtime logs during chat"),
):
    """Interact with the agent directly."""
    from finclaw.config.loader import load_config
    from finclaw.bus.queue import MessageBus
    from finclaw.agent.loop import AgentLoop
    from loguru import logger
    
    config = load_config()
    
    bus = MessageBus()
    provider = _make_provider(config)
    inner_provider = _make_inner_provider(config)

    if logs:
        logger.enable("finclaw")
    else:
        logger.disable("finclaw")

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        inner_model=config.agents.defaults.inner_model,
        inner_provider=inner_provider,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        search_api_key=config.tools.web.search.api_key or None,
        search_provider=config.tools.web.search.provider,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
    )
    
    # Show spinner when logs are off (no output to miss); skip when logs are on
    def _thinking_ctx():
        if logs:
            from contextlib import nullcontext
            return nullcontext()
        # Animated spinner is safe to use with prompt_toolkit input handling
        return console.status("[dim]FinClaw is thinking...[/dim]", spinner="dots")

    if message:
        # Single message mode
        async def run_once():
            with _thinking_ctx():
                response = await agent_loop.process_direct(message, session_id)
            _print_agent_response(response, render_markdown=markdown)
            await agent_loop.close_mcp()
        
        asyncio.run(run_once())
    else:
        # Interactive mode
        _init_prompt_session()
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

        def _exit_on_sigint(signum, frame):
            _restore_terminal()
            console.print("\nGoodbye!")
            os._exit(0)

        signal.signal(signal.SIGINT, _exit_on_sigint)
        
        async def run_interactive():
            try:
                while True:
                    try:
                        _flush_pending_tty_input()
                        user_input = await _read_interactive_input_async()
                        command = user_input.strip()
                        if not command:
                            continue

                        if _is_exit_command(command):
                            _restore_terminal()
                            console.print("\nGoodbye!")
                            break
                        
                        with _thinking_ctx():
                            response = await agent_loop.process_direct(user_input, session_id)
                        _print_agent_response(response, render_markdown=markdown)
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                await agent_loop.close_mcp()
        
        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from finclaw.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")
    table.add_column("Configuration", style="yellow")

    # WhatsApp
    wa = config.channels.whatsapp
    table.add_row(
        "WhatsApp",
        "✓" if wa.enabled else "✗",
        wa.bridge_url
    )

    dc = config.channels.discord
    table.add_row(
        "Discord",
        "✓" if dc.enabled else "✗",
        dc.gateway_url
    )

    # Feishu
    fs = config.channels.feishu
    fs_config = f"app_id: {fs.app_id[:10]}..." if fs.app_id else "[dim]not configured[/dim]"
    table.add_row(
        "Feishu",
        "✓" if fs.enabled else "✗",
        fs_config
    )

    # Mochat
    mc = config.channels.mochat
    mc_base = mc.base_url or "[dim]not configured[/dim]"
    table.add_row(
        "Mochat",
        "✓" if mc.enabled else "✗",
        mc_base
    )
    
    # Telegram
    tg = config.channels.telegram
    tg_config = f"token: {tg.token[:10]}..." if tg.token else "[dim]not configured[/dim]"
    table.add_row(
        "Telegram",
        "✓" if tg.enabled else "✗",
        tg_config
    )

    # Slack
    slack = config.channels.slack
    slack_config = "socket" if slack.app_token and slack.bot_token else "[dim]not configured[/dim]"
    table.add_row(
        "Slack",
        "✓" if slack.enabled else "✗",
        slack_config
    )

    console.print(table)


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess
    
    # User's bridge location
    user_bridge = Path.home() / ".finclaw" / "bridge"
    
    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge
    
    # Check for npm
    if not shutil.which("npm"):
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)
    
    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # finclaw/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)
    
    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge
    
    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall finclaw")
        raise typer.Exit(1)
    
    console.print(f"{__logo__} Setting up bridge...")
    
    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))
    
    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=user_bridge, check=True, capture_output=True)
        
        console.print("  Building...")
        subprocess.run(["npm", "run", "build"], cwd=user_bridge, check=True, capture_output=True)
        
        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)
    
    return user_bridge


@channels_app.command("login")
def channels_login():
    """Link device via QR code."""
    import subprocess
    from finclaw.config.loader import load_config
    
    config = load_config()
    bridge_dir = _get_bridge_dir()
    
    console.print(f"{__logo__} Starting bridge...")
    console.print("Scan the QR code to connect.\n")
    
    env = {**os.environ}
    if config.channels.whatsapp.bridge_token:
        env["BRIDGE_TOKEN"] = config.channels.whatsapp.bridge_token
    
    try:
        subprocess.run(["npm", "start"], cwd=bridge_dir, check=True, env=env)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Bridge failed: {e}[/red]")
    except FileNotFoundError:
        console.print("[red]npm not found. Please install Node.js.[/red]")


# ============================================================================
# Cron Commands
# ============================================================================

cron_app = typer.Typer(help="Manage scheduled tasks")
app.add_typer(cron_app, name="cron")


@cron_app.command("list")
def cron_list(
    all: bool = typer.Option(False, "--all", "-a", help="Include disabled jobs"),
):
    """List scheduled jobs."""
    from finclaw.config.loader import get_data_dir
    from finclaw.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    jobs = service.list_jobs(include_disabled=all)
    
    if not jobs:
        console.print("No scheduled jobs.")
        return
    
    table = Table(title="Scheduled Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Schedule")
    table.add_column("Status")
    table.add_column("Next Run")
    
    import time
    for job in jobs:
        # Format schedule
        if job.schedule.kind == "every":
            sched = f"every {(job.schedule.every_ms or 0) // 1000}s"
        elif job.schedule.kind == "cron":
            sched = job.schedule.expr or ""
        else:
            sched = "one-time"
        
        # Format next run
        next_run = ""
        if job.state.next_run_at_ms:
            next_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(job.state.next_run_at_ms / 1000))
            next_run = next_time
        
        status = "[green]enabled[/green]" if job.enabled else "[dim]disabled[/dim]"
        
        table.add_row(job.id, job.name, sched, status, next_run)
    
    console.print(table)


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", "-n", help="Job name"),
    message: str = typer.Option(..., "--message", "-m", help="Message for agent"),
    every: int = typer.Option(None, "--every", "-e", help="Run every N seconds"),
    cron_expr: str = typer.Option(None, "--cron", "-c", help="Cron expression (e.g. '0 9 * * *')"),
    at: str = typer.Option(None, "--at", help="Run once at time (ISO format)"),
    deliver: bool = typer.Option(False, "--deliver", "-d", help="Deliver response to channel"),
    to: str = typer.Option(None, "--to", help="Recipient for delivery"),
    channel: str = typer.Option(None, "--channel", help="Channel for delivery (e.g. 'telegram', 'whatsapp')"),
):
    """Add a scheduled job."""
    from finclaw.config.loader import get_data_dir
    from finclaw.cron.service import CronService
    from finclaw.cron.types import CronSchedule
    
    # Determine schedule type
    if every:
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
    elif cron_expr:
        schedule = CronSchedule(kind="cron", expr=cron_expr)
    elif at:
        import datetime
        dt = datetime.datetime.fromisoformat(at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        console.print("[red]Error: Must specify --every, --cron, or --at[/red]")
        raise typer.Exit(1)
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    job = service.add_job(
        name=name,
        schedule=schedule,
        message=message,
        deliver=deliver,
        to=to,
        channel=channel,
    )
    
    console.print(f"[green]✓[/green] Added job '{job.name}' ({job.id})")


@cron_app.command("remove")
def cron_remove(
    job_id: str = typer.Argument(..., help="Job ID to remove"),
):
    """Remove a scheduled job."""
    from finclaw.config.loader import get_data_dir
    from finclaw.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    if service.remove_job(job_id):
        console.print(f"[green]✓[/green] Removed job {job_id}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("enable")
def cron_enable(
    job_id: str = typer.Argument(..., help="Job ID"),
    disable: bool = typer.Option(False, "--disable", help="Disable instead of enable"),
):
    """Enable or disable a job."""
    from finclaw.config.loader import get_data_dir
    from finclaw.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    job = service.enable_job(job_id, enabled=not disable)
    if job:
        status = "disabled" if disable else "enabled"
        console.print(f"[green]✓[/green] Job '{job.name}' {status}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("run")
def cron_run(
    job_id: str = typer.Argument(..., help="Job ID to run"),
    force: bool = typer.Option(False, "--force", "-f", help="Run even if disabled"),
):
    """Manually run a job."""
    from finclaw.config.loader import get_data_dir
    from finclaw.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    async def run():
        return await service.run_job(job_id, force=force)
    
    if asyncio.run(run()):
        console.print(f"[green]✓[/green] Job executed")
    else:
        console.print(f"[red]Failed to run job {job_id}[/red]")


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status():
    """Show finclaw status."""
    from finclaw.config.loader import load_config, get_config_path

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} finclaw Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from finclaw.providers.registry import PROVIDERS

        console.print(f"Model: {config.agents.defaults.model}")
        
        # Check API keys from registry
        for spec in PROVIDERS:
            p = getattr(config.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_local:
                # Local deployments show api_base instead of api_key
                if p.api_base:
                    console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(p.api_key)
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")

        # Financial data keys
        fred_key = config.tools.financial.fred_api_key
        console.print(f"FRED API: {'[green]✓[/green]' if fred_key else '[dim]not set[/dim] (optional — set tools.financial.fredApiKey for macro data)'}")

        # Meme monitor keys
        meme_cfg = config.tools.meme_monitor
        has_cookies = bool(meme_cfg.twitter_cookies)
        has_rsshub = bool(meme_cfg.rsshub_base_url)
        console.print(f"Twitter Cookies: {'[green]✓[/green]' if has_cookies else '[dim]not set[/dim] (optional — set tools.memeMonitor.twitterCookies)'}")
        console.print(f"RSSHub Base URL: {'[green]✓ ' + meme_cfg.rsshub_base_url + '[/green]' if has_rsshub else '[dim]not set[/dim] (optional — set tools.memeMonitor.rsshubBaseUrl)'}")

        # Solana / meme creation
        has_sol_key = bool(meme_cfg.solana_private_key or os.environ.get("SOLANA_PRIVATE_KEY"))
        has_sol_rpc = bool(meme_cfg.solana_rpc_url or os.environ.get("SOLANA_RPC_URL"))
        console.print(f"Solana Key: {'[green]✓[/green]' if has_sol_key else '[dim]not set[/dim] (optional — set tools.memeMonitor.solanaPrivateKey)'}")
        console.print(f"Solana RPC: {'[green]✓[/green]' if has_sol_rpc else '[dim]not set[/dim] (optional — set tools.memeMonitor.solanaRpcUrl)'}")

        # BSC / four.meme creation
        has_bsc_key = bool(meme_cfg.bsc_private_key or os.environ.get("BSC_PRIVATE_KEY"))
        has_bsc_rpc = bool(meme_cfg.bsc_rpc_url or os.environ.get("BSC_RPC_URL"))
        console.print(f"BSC Key: {'[green]✓[/green]' if has_bsc_key else '[dim]not set[/dim] (optional — set tools.memeMonitor.bscPrivateKey)'}")
        console.print(f"BSC RPC: {'[green]✓[/green]' if has_bsc_rpc else '[dim]not set[/dim] (optional — set tools.memeMonitor.bscRpcUrl)'}")


# ============================================================================
# OAuth Login
# ============================================================================

provider_app = typer.Typer(help="Manage providers")
app.add_typer(provider_app, name="provider")


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="OAuth provider to authenticate with (e.g., 'openai-codex')"),
):
    """Authenticate with an OAuth provider."""
    console.print(f"{__logo__} OAuth Login - {provider}\n")

    if provider == "openai-codex":
        try:
            from oauth_cli_kit import get_token, login_oauth_interactive
            token = None
            try:
                token = get_token()
            except Exception:
                token = None
            if not (token and token.access):
                console.print("[cyan]No valid token found. Starting interactive OAuth login...[/cyan]")
                console.print("A browser window may open for you to authenticate.\n")
                token = login_oauth_interactive(
                    print_fn=lambda s: console.print(s),
                    prompt_fn=lambda s: typer.prompt(s),
                )
            if not (token and token.access):
                console.print("[red]✗ Authentication failed[/red]")
                raise typer.Exit(1)
            console.print("[green]✓ Successfully authenticated with OpenAI Codex![/green]")
            console.print(f"[dim]Account ID: {token.account_id}[/dim]")
        except ImportError:
            console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Authentication error: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]")
        console.print("[yellow]Supported providers: openai-codex[/yellow]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
