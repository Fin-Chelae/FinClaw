---
name: meme-create
description: Deploy a meme coin on pump.fun — collect required fields then call the meme tool.
---

# Meme Coin Creation

Use this skill when the user wants to **deploy/launch/create/mint** a meme coin (not just browse, search, or brainstorm).

## Step 1 — Collect Required Fields

Before calling `meme` for creation, you must have ALL of the following confirmed by the user:

| Field | Example | Notes |
|-------|---------|-------|
| `name` | Moon Cat | Full token name |
| `symbol` | MCAT | Ticker, uppercase, max 10 chars |
| `description` | A lunar explorer cat on Solana | 1–2 sentences |
| `image_path` | /Users/bella/mooncat.png | Absolute path to PNG/JPG/GIF on the user's machine |
| `platform` | pump.fun | Default: pump.fun (Solana). BSC not yet available. |

Optional fields (only ask if the user mentions them):
- `twitter` — Twitter/X URL
- `telegram` — Telegram URL
- `website` — Website URL
- `buy_amount` — Initial SOL buy (default: 0.01 SOL)

Ask for any missing required fields naturally in conversation. Do **not** call `meme` until `name`, `symbol`, `description`, and `image_path` are all in hand.

If the user has no image ready, ask them to provide one (or a placeholder) before proceeding — `image_path` is required by pump.fun.

## Step 2 — Confirm with the User

Before deploying, summarise and confirm:

> "Ready to deploy **Moon Cat (MCAT)** on pump.fun with a 0.01 SOL initial buy. Logo: mooncat.png. Shall I proceed?"

## Step 3 — Call meme with a packed query

Pack all confirmed fields into a single query string:

```
meme(query="Create token: name='Moon Cat', symbol='MCAT', description='A lunar explorer cat meme coin on Solana', image_path='/Users/bella/mooncat.png', platform='pump.fun'")
```

With optional social links:
```
meme(query="Create token: name='Moon Cat', symbol='MCAT', description='A lunar explorer cat', image_path='/path/to/img.png', platform='pump.fun', twitter='https://x.com/mooncat', telegram='https://t.me/mooncat'")
```

The inner agent handles environment checks (`check_env`) and deployment automatically — do not add those steps yourself.

## After Deployment

A successful result includes:
- `mint` — token contract address
- `pump_fun_url` — `https://pump.fun/<mint>`
- `solscan_url` — transaction link on Solscan

Share both links with the user.
