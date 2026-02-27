<p align="center">
  <img src="assets/fin-claw.jpg" alt="FinClaw" width="600"/>
</p>
  <p align="center">
    <Strong>Your AI-powered financial analyst — multi-market, multi-channel, always on.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/LLM-14%2B%20providers-purple" alt="14+ LLM Providers">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
    <img src="https://img.shields.io/badge/status-alpha-orange" alt="Alpha">
  </p>
</p>

---

## What is FinClaw?

FinClaw is an **AI financial agent** that connects to live market data, reasons through investment questions, and delivers analysis wherever you are: Telegram, Discord, Slack, email, or the command line.
 It covers US/global equities (Yahoo Finance), Chinese A-shares (AKShare), macro indicators (FRED), Crypto (DexScreener + CoinGecko + social scanning), and prediction markets (Polymarket + Kalshi), then combines tools + LLM reasoning into an auditable research workflow.

FinClaw can also **launch meme coins end-to-end**: scan Twitter and RSS feeds for viral ideas, evaluate meme potential with LLM scoring, and deploy tokens directly on **pump.fun (Solana)** or **four.meme (BSC)** — all from a single natural-language prompt.

<img src="assets/fin-claw-infographic.svg" alt="FinClaw Infographic" width="800"/>

## Why FinClaw

| | |
|---|---|
| **Multi-Market Coverage** | US/global equities (Yahoo Finance), Chinese A-shares (AKShare), macro data (FRED), crypto (DexScreener + CoinGecko), prediction markets (Polymarket + Kalshi) |
| **9 Chat Channels** | Telegram, Discord, Slack, WhatsApp, Feishu, DingTalk, Email, QQ, CLI |
| **14+ LLM Providers** | Anthropic, OpenAI, Gemini, DeepSeek, Groq, OpenRouter, and more via LiteLLM |
| **Agentic Tool Loop** | Autonomous multi-step reasoning with up to 20 tool iterations per query |
| **Financial Memory** | Tracks your investment profile, analysis history, and data cache with TTL |
| **Scheduled Reports** | Built-in cron system for recurring market briefs and alerts |
| **Meme Coin Launch** | Scan social signals, evaluate ideas, and deploy tokens on pump.fun (Solana) or four.meme (BSC) via natural language |
| **Extensible Skills** | Plugin architecture; add custom skills or connect MCP servers |
| **Production-Ready** | Docker support, async event loop, per-channel session management |

## Roadmap

- [ ] **Automated meme pipeline** — end-to-end automation: monitor social feeds continuously, auto-score viral candidates, and optionally deploy tokens when signals cross a configurable threshold.

## Outcome-Driven Use Cases

- **Daily market brief**: run fixed-time summaries to monitor indices, rates, and watchlist names.
- **Thesis tracking**: revisit prior analysis and detect whether the core thesis is strengthening or breaking.
- **Cross-market comparison**: compare US and CN peers with the same prompt pattern.
- **Prediction market intelligence**: query Polymarket and Kalshi for event odds, compare cross-platform probabilities, chart probability history over time, and surface high-volume markets as early signals.
- **Team analyst copilot**: run one shared logic layer across Slack/Discord/Telegram for internal research.
- **Meme coin launch**: scan social feeds for viral ideas, score them, and deploy a token on pump.fun or four.meme in one conversation.

## Quick Start

### Install

```bash
pip install -e .
```

### Set up

```bash
finclaw onboard
# Edit ~/.finclaw/config.json to add your LLM API key
```

### Run

```bash
# Interactive chat
finclaw agent

# Single question
finclaw agent -m "Analyze AAPL's latest earnings and valuation"

# Start multi-channel gateway
finclaw gateway -p 18790
```

### Docker

```bash
# Build
docker build -t finclaw .

# Gateway
docker run -d \
  -v ~/.finclaw:/root/.finclaw \
  -p 18790:18790 \
  finclaw gateway

# One-off query
docker run --rm \
  -v ~/.finclaw:/root/.finclaw \
  finclaw agent -m "What's the current US CPI trend?"
```

## Configuration

FinClaw stores config at `~/.finclaw/config.json`.

```jsonc
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "max_tokens": 8192,
      "temperature": 0.7,
      "max_tool_iterations": 20,
      "logVerbose": false        // set true to log full LLM request/response payloads
    }
  },
  "providers": {
    "anthropic": { "apiKey": "sk-ant-..." },
    "openai":    { "apiKey": "sk-..." }
  },
  "channels": {
    "telegram": { "enabled": true, "token": "BOT_TOKEN", "allowFrom": ["user_id"] }
  },
  "tools": {
    "web": { "search": { "provider": "tavily", "apiKey": "tvly-..." } },
    "financial": { "fredApiKey": "your-fred-key" },
    "memeMonitor": {
      "twitterCookies": { "authToken": "", "ct0": "" },
      "coingeckoApiKey": "",
      "rsshubBaseUrl": "",
      "solanaPrivateKey": "",
      "solanaRpcUrl": "",
      "bscPrivateKey": "",
      "bscRpcUrl": ""
    }
  }
}
```

Run `finclaw onboard` to generate starter config interactively.

## Data Source API Keys

Most data sources work out of the box, but some require API keys for full functionality.

| Key | Required? | How to Get | Purpose |
|-----|-----------|------------|---------|
| **FRED API key** | Recommended | Free at [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/api_key.html) | US macro data (GDP, CPI, unemployment, yields) |
| **CoinGecko API key** | Optional | Free tier at [coingecko.com/api](https://www.coingecko.com/en/api) | Higher rate limits for meme coin metadata (30 -> 500 req/min) |
| **Twitter cookies** | Optional | Extract `auth_token` and `ct0` from browser DevTools after logging into X/Twitter | Native Twitter scanning for meme coin social signals |
| **RSSHub base URL** | Optional | Self-host or use a public instance ([docs.rsshub.app](https://docs.rsshub.app/)) | RSS-based social scanning (Twitter, Reddit, TikTok feeds) |
| **Solana private key** | For pump.fun launch | Your Solana wallet private key (base58, hex, or JSON byte-array) | Signing pump.fun token-creation transactions |
| **Solana RPC URL** | For pump.fun launch | Public: `https://api.mainnet-beta.solana.com` or a paid RPC | Broadcasting signed transactions to Solana |
| **BSC private key** | For four.meme launch | Your BSC wallet private key (hex, with or without 0x prefix) | Signing four.meme token-creation transactions on BSC |
| **BSC RPC URL** | For four.meme launch | Default: `https://bsc-dataseed.binance.org` or a paid RPC | Broadcasting signed transactions to BSC |

Set these in `~/.finclaw/config.json`:

```jsonc
{
  "tools": {
    // Macro data — without this key, FRED commands return an error
    "financial": { "fredApiKey": "your-fred-key" },

    // Meme coin tools — all fields optional
    "memeMonitor": {
      "twitterCookies": { "authToken": "...", "ct0": "..." },
      "coingeckoApiKey": "CG-...",
      "rsshubBaseUrl": "https://rsshub.example.com",
      "solanaPrivateKey": "<base58-or-hex-key>",
      "solanaRpcUrl": "https://api.mainnet-beta.solana.com",
      "bscPrivateKey": "<hex-key>",
      "bscRpcUrl": "https://bsc-dataseed.binance.org"
    }
  }
}
```

Wallet credentials can also be set as environment variables (`SOLANA_PRIVATE_KEY`, `SOLANA_RPC_URL`, `BSC_PRIVATE_KEY`, `BSC_RPC_URL`) — config takes priority, env vars serve as fallback.


## Meme Coin Launch

FinClaw can deploy a new token on **pump.fun (Solana)** or **four.meme (BSC)** from a single prompt.

### How it works

1. **Scan** — `launch_scan` searches Twitter and RSS feeds for viral ideas, extracts meme-worthy keywords with LLM scoring, and ranks candidates.
2. **Confirm** — FinClaw presents candidates and waits for your approval before proceeding.
3. **Check env** — verifies wallet credentials are configured for the target platform.
4. **Deploy** — uploads token image + metadata, builds and signs the on-chain transaction, and broadcasts it.
5. **Result** — returns the platform URL and block explorer transaction link.

| Platform | Chain | Fee | What FinClaw does |
|----------|-------|-----|-------------------|
| **pump.fun** | Solana | SOL gas + optional initial buy | Upload to IPFS, build Solana tx via PumpPortal, sign with `solders`, broadcast via RPC |
| **four.meme** | BSC | 0.001 BNB + optional presale | Login via signature, upload image, get `createArg` from API, call `TokenManager2.createToken` on-chain |

### Setup

Wallet credentials go in `~/.finclaw/config.json` under `tools.memeMonitor`:

```jsonc
{
  "tools": {
    "memeMonitor": {
      // pump.fun (Solana)
      "solanaPrivateKey": "<base58-or-hex-key>",
      "solanaRpcUrl": "https://api.mainnet-beta.solana.com",

      // four.meme (BSC)
      "bscPrivateKey": "<hex-key>",
      "bscRpcUrl": "https://bsc-dataseed.binance.org"
    }
  }
}
```

You can verify the setup at any time:

```
> Check if my wallet is ready to launch a meme coin
```

### Example prompts

```
> Scan social media for meme coin launch ideas and suggest the top candidate
> Launch a meme coin called CLAW with symbol CLAW on pump.fun.
  Image is at /path/to/claw.png. Initial buy: 0.05 SOL.
> Create a token called BiBiLaBu (BBLB) on four.meme. Image: /path/to/bblb.png. Label: Meme.
> What's the pump.fun URL for my last launch?
```

> [!WARNING]
> Token creation is irreversible and costs real SOL/BNB (gas + optional initial buy/presale).
> FinClaw always confirms token details with you before signing. Never share your private key.

## Chat Apps

Talk to FinClaw through Telegram, Discord, WhatsApp, Feishu, Mochat, DingTalk, Slack, Email, or QQ — anytime, anywhere.

| Channel | Setup |
|---------|-------|
| **Telegram** | Easy (just a token) |
| **Discord** | Easy (bot token + intents) |
| **WhatsApp** | Medium (scan QR) |
| **Feishu** | Medium (app credentials) |
| **Mochat** | Medium (claw token + websocket) |
| **DingTalk** | Medium (app credentials) |
| **Slack** | Medium (bot + app tokens) |
| **Email** | Medium (IMAP/SMTP credentials) |
| **QQ** | Easy (app credentials) |

<details>
<summary><b>Telegram</b> (Recommended)</summary>

**1. Create a bot**
- Open Telegram, search `@BotFather`
- Send `/newbot`, follow prompts
- Copy the token

**2. Configure**

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

> You can find your **User ID** in Telegram settings. It is shown as `@yourUserId`.
> Copy this value **without the `@` symbol** and paste it into the config file.


**3. Run**

```bash
finclaw gateway
```

</details>

<details>
<summary><b>Mochat (Claw IM)</b></summary>

Uses **Socket.IO WebSocket** by default, with HTTP polling fallback.

**1. Ask FinClaw to set up Mochat for you**

Simply send this message to FinClaw (replace `xxx@xxx` with your real email):

```
Read https://raw.githubusercontent.com/HKUDS/MoChat/refs/heads/main/skills/nanobot/skill.md and register on MoChat. My Email account is xxx@xxx Bind me as your owner and DM me on MoChat.
```

FinClaw will automatically register, configure `~/.finclaw/config.json`, and connect to Mochat.

**2. Restart gateway**

```bash
finclaw gateway
```

That's it — FinClaw handles the rest!

<br>

<details>
<summary>Manual configuration (advanced)</summary>

If you prefer to configure manually, add the following to `~/.finclaw/config.json`:

> Keep `claw_token` private. It should only be sent in `X-Claw-Token` header to your Mochat API endpoint.

```json
{
  "channels": {
    "mochat": {
      "enabled": true,
      "base_url": "https://mochat.io",
      "socket_url": "https://mochat.io",
      "socket_path": "/socket.io",
      "claw_token": "claw_xxx",
      "agent_user_id": "6982abcdef",
      "sessions": ["*"],
      "panels": ["*"],
      "reply_delay_mode": "non-mention",
      "reply_delay_ms": 120000
    }
  }
}
```

</details>

</details>

<details>
<summary><b>Discord</b></summary>

**1. Create a bot**
- Go to https://discord.com/developers/applications
- Create an application -> Bot -> Add Bot
- Copy the bot token

**2. Enable intents**
- In the Bot settings, enable **MESSAGE CONTENT INTENT**
- (Optional) Enable **SERVER MEMBERS INTENT** if you plan to use allow lists based on member data

**3. Get your User ID**
- Discord Settings -> Advanced -> enable **Developer Mode**
- Right-click your avatar -> **Copy User ID**

**4. Configure**

```json
{
  "channels": {
    "discord": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

**5. Invite the bot**
- OAuth2 -> URL Generator
- Scopes: `bot`
- Bot Permissions: `Send Messages`, `Read Message History`
- Open the generated invite URL and add the bot to your server

**6. Run**

```bash
finclaw gateway
```

</details>

<details>
<summary><b>WhatsApp</b></summary>

Requires **Node.js >= 18**.

**1. Link device**

```bash
finclaw channels login
# Scan QR with WhatsApp -> Settings -> Linked Devices
```

**2. Configure**

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

**3. Run** (two terminals)

```bash
# Terminal 1
finclaw channels login

# Terminal 2
finclaw gateway
```

</details>

<details>
<summary><b>Feishu</b></summary>

Uses **WebSocket** long connection — no public IP required.

**1. Create a Feishu bot**
- Visit [Feishu Open Platform](https://open.feishu.cn/app)
- Create a new app -> Enable **Bot** capability
- **Permissions**: Add `im:message` (send messages)
- **Events**: Add `im.message.receive_v1` (receive messages)
  - Select **Long Connection** mode
- Get **App ID** and **App Secret** from "Credentials & Basic Info"
- Publish the app

**2. Configure**

```json
{
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "cli_xxx",
      "appSecret": "xxx",
      "encryptKey": "",
      "verificationToken": "",
      "allowFrom": []
    }
  }
}
```

> `encryptKey` and `verificationToken` are optional for Long Connection mode.
> `allowFrom`: Leave empty to allow all users, or add `["ou_xxx"]` to restrict access.

**3. Run**

```bash
finclaw gateway
```

> [!TIP]
> Feishu uses WebSocket to receive messages — no webhook or public IP needed!

</details>

<details>
<summary><b>QQ</b></summary>

Uses **botpy SDK** with WebSocket — no public IP required. Currently supports **private messages only**.

**1. Register & create bot**
- Visit [QQ Open Platform](https://q.qq.com) -> Register as a developer (personal or enterprise)
- Create a new bot application
- Go to **Developer Settings** -> copy **AppID** and **AppSecret**

**2. Set up sandbox for testing**
- In the bot management console, find **Sandbox Config**
- Add your own QQ number as a test member
- Scan the bot's QR code with mobile QQ -> open the bot profile -> tap "Send Message" to start chatting

**3. Configure**

> - `allowFrom`: Leave empty for public access, or add user openids to restrict. You can find openids in the FinClaw logs when a user messages the bot.
> - For production: submit a review in the bot console and publish. See [QQ Bot Docs](https://bot.q.qq.com/wiki/) for the full publishing flow.

```json
{
  "channels": {
    "qq": {
      "enabled": true,
      "appId": "YOUR_APP_ID",
      "secret": "YOUR_APP_SECRET",
      "allowFrom": []
    }
  }
}
```

**4. Run**

```bash
finclaw gateway
```

Now send a message to the bot from QQ — it should respond!

</details>

<details>
<summary><b>DingTalk</b></summary>

Uses **Stream Mode** — no public IP required.

**1. Create a DingTalk bot**
- Visit [DingTalk Open Platform](https://open-dev.dingtalk.com/)
- Create a new app -> Add **Robot** capability
- **Configuration**:
  - Toggle **Stream Mode** ON
- **Permissions**: Add necessary permissions for sending messages
- Get **AppKey** (Client ID) and **AppSecret** (Client Secret) from "Credentials"
- Publish the app

**2. Configure**

```json
{
  "channels": {
    "dingtalk": {
      "enabled": true,
      "clientId": "YOUR_APP_KEY",
      "clientSecret": "YOUR_APP_SECRET",
      "allowFrom": []
    }
  }
}
```

> `allowFrom`: Leave empty to allow all users, or add `["staffId"]` to restrict access.

**3. Run**

```bash
finclaw gateway
```

</details>

<details>
<summary><b>Slack</b></summary>

Uses **Socket Mode** — no public URL required.

**1. Create a Slack app**
- Go to [Slack API](https://api.slack.com/apps) -> **Create New App** -> "From scratch"
- Pick a name and select your workspace

**2. Configure the app**
- **Socket Mode**: Toggle ON -> Generate an **App-Level Token** with `connections:write` scope -> copy it (`xapp-...`)
- **OAuth & Permissions**: Add bot scopes: `chat:write`, `reactions:write`, `app_mentions:read`
- **Event Subscriptions**: Toggle ON -> Subscribe to bot events: `message.im`, `message.channels`, `app_mention` -> Save Changes
- **App Home**: Scroll to **Show Tabs** -> Enable **Messages Tab** -> Check **"Allow users to send Slash commands and messages from the messages tab"**
- **Install App**: Click **Install to Workspace** -> Authorize -> copy the **Bot Token** (`xoxb-...`)

**3. Configure FinClaw**

```json
{
  "channels": {
    "slack": {
      "enabled": true,
      "botToken": "xoxb-...",
      "appToken": "xapp-...",
      "groupPolicy": "mention"
    }
  }
}
```

**4. Run**

```bash
finclaw gateway
```

DM the bot directly or @mention it in a channel — it should respond!

> [!TIP]
> - `groupPolicy`: `"mention"` (default — respond only when @mentioned), `"open"` (respond to all channel messages), or `"allowlist"` (restrict to specific channels).
> - DM policy defaults to open. Set `"dm": {"enabled": false}` to disable DMs.

</details>

<details>
<summary><b>Email</b></summary>

Give FinClaw its own email account. It polls **IMAP** for incoming mail and replies via **SMTP** — like a personal email assistant.

**1. Get credentials (Gmail example)**
- Create a dedicated Gmail account for your bot (e.g. `finclaw-bot@gmail.com`)
- Enable 2-Step Verification -> Create an [App Password](https://myaccount.google.com/apppasswords)
- Use this app password for both IMAP and SMTP

**2. Configure**

> - `consentGranted` must be `true` to allow mailbox access. This is a safety gate — set `false` to fully disable.
> - `allowFrom`: Leave empty to accept emails from anyone, or restrict to specific senders.
> - `smtpUseTls` and `smtpUseSsl` default to `true` / `false` respectively, which is correct for Gmail (port 587 + STARTTLS). No need to set them explicitly.
> - Set `"autoReplyEnabled": false` if you only want to read/analyze emails without sending automatic replies.

```json
{
  "channels": {
    "email": {
      "enabled": true,
      "consentGranted": true,
      "imapHost": "imap.gmail.com",
      "imapPort": 993,
      "imapUsername": "finclaw-bot@gmail.com",
      "imapPassword": "your-app-password",
      "smtpHost": "smtp.gmail.com",
      "smtpPort": 587,
      "smtpUsername": "finclaw-bot@gmail.com",
      "smtpPassword": "your-app-password",
      "fromAddress": "finclaw-bot@gmail.com",
      "allowFrom": ["your-real-email@gmail.com"]
    }
  }
}
```


**3. Run**

```bash
finclaw gateway
```

</details>


## Supported Data Sources

| Source | Coverage             | Data |
|--------|----------------------|------|
| **Yahoo Finance** | US & global equities | Real-time quotes, OHLCV, financials, analyst estimates, insider trades |
| **AKShare** | Chinese A-shares     | Quotes, K-line history, financial reports, sector rankings, index data |
| **FRED** | US macroeconomic     | GDP, CPI, unemployment, Treasury yields, Fed funds rate, M2, and more |
| **DexScreener** | DEX tokens (all chains) | Pair search, price, volume, liquidity, boosted/trending tokens |
| **CoinGecko** | Crypto / meme coins  | Trending coins, coin metadata, market cap, price aggregation |
| **Tavily / Brave** | Web Search           | Financial news search, company filings, general web lookup |
| **Twitter / X** | Social signals       | KOL tweet scanning, meme word extraction, viral content detection |
| **RSS (via RSSHub)** | Multi-platform feeds | Reddit, TikTok, Truth Social, and custom feeds for meme monitoring |
| **Polymarket** | Prediction markets (Polymarket) | Trending markets, keyword search, event odds, probability history, cross-platform comparison, probability charts |
| **Kalshi** | Prediction markets (Kalshi) | Trending markets, keyword search, event and market detail, cross-platform comparison |

## Supported Channels

| Channel | Protocol | Key Config |
|---------|----------|------------|
| **Telegram** | Bot API (polling) | `token`, `allow_from` |
| **Discord** | Gateway (WebSocket) | `token`, `allow_from` |
| **Slack** | Socket Mode | `bot_token`, `app_token` |
| **WhatsApp** | Node.js bridge (Baileys) | `bridge_url` |
| **Feishu / Lark** | WebSocket | `app_id`, `app_secret` |
| **DingTalk** | Stream | `client_id`, `client_secret` |
| **Email** | IMAP + SMTP | Host, credentials, polling interval |
| **QQ** | Bot SDK | `app_id`, `secret` |
| **CLI** | Interactive terminal | — |

All channels share one interface; enable any combination in `config.json`.

## Architecture

<p align="center">
  <img src="assets/architecture.jpg" alt="FinClaw Architecture" width="800"/>
</p>

1. **Chat Apps** — messages from 9 channels (Telegram, Discord, Slack, WhatsApp, etc.) enter the async **Message Bus**.
2. **Agent Loop** — the LLM reasons, calls the **Financial Tool Router** (financial metrics, economics, web, cron), and evaluates its own **Internal Response** in a reflect-and-retry cycle (up to 20 iterations).
3. **Customized Agent & Personal Profiling** — each response is shaped by the agent's **Soul** (personality), **Memory** (long-term facts), **Financial Profile** (investment preferences), and **Chat & Analysis History**.
4. **Final Response** — once approved through the profiling layer, the response routes back to the originating channel.

## Project Structure

```text
finclaw/
├── agent/
│   ├── loop.py                 # Core agent reasoning loop
│   ├── context.py              # System prompt assembly
│   ├── memory.py               # Long-term memory store
│   ├── financial/              # Financial specialization
│   │   ├── intent.py           #   Query intent detection
│   │   ├── profile.py          #   Investment profile management
│   │   ├── router.py           #   Metrics & search routing
│   │   ├── meme_router.py      #   Meme coin unified router
│   │   ├── prediction_market_router.py  # Prediction market sub-agent
│   │   ├── history.py          #   Analysis history tracking
│   │   └── cache.py            #   Data cache with TTL
│   ├── financial_tools/        # Data source integrations
│   │   ├── yfinance_tool.py    #   US/global equities
│   │   ├── akshare_tool.py     #   Chinese A-shares
│   │   ├── economics_data_tool.py  # FRED macro data
│   │   ├── meme/               #   Meme coin pipeline
│   │   │   ├── meme_data_tool.py   # DexScreener + CoinGecko
│   │   │   ├── meme_search_tool.py # Twitter + RSS scanning
│   │   │   └── meme_create_tool.py # Token deployment (pump.fun + four.meme)
│   │   └── prediction_market/  #   Prediction market pipeline
│   │       ├── prediction_market_data_tool.py  # Polymarket + Kalshi APIs
│   │       └── prediction_market_tool.py       # Command dispatch
│   └── tools/                  # General-purpose tools
│       ├── web.py              #   Web search & fetch
│       ├── filesystem.py       #   File operations
│       ├── shell.py            #   Shell execution
│       ├── cron.py             #   Scheduled tasks
│       └── mcp.py              #   MCP server connections
├── channels/                   # Chat platform integrations
├── providers/                  # LLM provider implementations
├── config/                     # Pydantic config schema & loader
├── bus/                        # Async message queue & events
├── skills/                     # Built-in skills
├── cli/commands.py             # CLI entry point (Typer)
└── bridge/                     # WhatsApp Node.js bridge
```

## Contributing

Contributions are welcome.

## License

[MIT](LICENSE)
