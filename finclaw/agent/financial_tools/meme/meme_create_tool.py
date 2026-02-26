"""Meme coin creation tool — deploy tokens on pump.fun (Solana) or four.meme (BSC).

Reads wallet credentials from config (tools.memeMonitor), falling back to
environment variables for backward compatibility:
  - Solana: solanaPrivateKey / SOLANA_PRIVATE_KEY, solanaRpcUrl / SOLANA_RPC_URL
  - BSC:    bscPrivateKey / BSC_PRIVATE_KEY,       bscRpcUrl / BSC_RPC_URL
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from uuid import uuid4

import asyncio
from loguru import logger

from finclaw.agent.tools.base import Tool

# Confirmation expiry timeout in seconds
CONFIRMATION_TIMEOUT_SEC = 300  # 5 minutes

# ---------------------------------------------------------------------------
# pump.fun constants
# ---------------------------------------------------------------------------
PUMPFUN_IPFS_URL = "https://pump.fun/api/ipfs"
PUMPPORTAL_API_URL = "https://pumpportal.fun/api/trade-local"
DEFAULT_BUY_AMOUNT = 0.01
DEFAULT_SLIPPAGE_BPS = 10
DEFAULT_PRIORITY_FEE = 0.0005

# ---------------------------------------------------------------------------
# four.meme (BSC) constants
# ---------------------------------------------------------------------------
FOUR_MEME_API = "https://four.meme/meme-api/v1"
TOKEN_MANAGER2 = "0x5c952063c7fc8610FFDB798152D69F0B9550762b"
CREATE_FEE_BNB = 0.001
DEFAULT_PRESALE_BNB = 0
VALID_LABELS = ["Meme", "AI", "Defi", "Games", "Infra", "De-Sci", "Social", "Depin", "Charity", "Others"]

TOKEN_MANAGER2_ABI = [
    {
        "inputs": [
            {"internalType": "bytes", "name": "createArg", "type": "bytes"},
            {"internalType": "bytes", "name": "sign", "type": "bytes"},
        ],
        "name": "createToken",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "name": "creator", "type": "address"},
            {"indexed": False, "name": "token", "type": "address"},
            {"indexed": False, "name": "requestId", "type": "uint256"},
            {"indexed": False, "name": "name", "type": "string"},
            {"indexed": False, "name": "symbol", "type": "string"},
            {"indexed": False, "name": "totalSupply", "type": "uint256"},
            {"indexed": False, "name": "launchTime", "type": "uint256"},
            {"indexed": False, "name": "launchFee", "type": "uint256"},
        ],
        "name": "TokenCreate",
        "type": "event",
    },
]


# =====================================================================
# Credential helpers
# =====================================================================

def _get_solana_credentials() -> tuple[str, str]:
    """Return (private_key, rpc_url) from config, falling back to env vars."""
    from finclaw.config.loader import load_config
    cfg = load_config().tools.meme_monitor
    private_key = cfg.solana_private_key or os.environ.get("SOLANA_PRIVATE_KEY", "")
    rpc_url = cfg.solana_rpc_url or os.environ.get("SOLANA_RPC_URL", "")
    return private_key, rpc_url


def _get_bsc_credentials() -> tuple[str, str]:
    """Return (private_key, rpc_url) from config, falling back to env vars."""
    from finclaw.config.loader import load_config
    cfg = load_config().tools.meme_monitor
    private_key = cfg.bsc_private_key or os.environ.get("BSC_PRIVATE_KEY", "")
    rpc_url = cfg.bsc_rpc_url or os.environ.get("BSC_RPC_URL", "https://bsc-dataseed.binance.org")
    return private_key, rpc_url


# =====================================================================
# pump.fun (Solana) helpers
# =====================================================================

def _upload_to_ipfs(image_path, name, symbol, description,
                    twitter="", telegram="", website=""):
    """Upload token image + metadata to pump.fun IPFS. Returns metadata URI."""
    import requests

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg",
                  ".jpeg": "image/jpeg", ".gif": "image/gif"}
    mime_type = mime_types.get(ext, "application/octet-stream")

    with open(image_path, "rb") as f:
        resp = requests.post(
            PUMPFUN_IPFS_URL,
            data={"name": name, "symbol": symbol, "description": description,
                  "twitter": twitter, "telegram": telegram,
                  "website": website, "showName": "true"},
            files={"file": (os.path.basename(image_path), f, mime_type)},
            timeout=30,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"IPFS upload failed (HTTP {resp.status_code}): {resp.text}")

    metadata_uri = resp.json().get("metadataUri")
    if not metadata_uri:
        raise RuntimeError(f"No metadataUri in response: {resp.json()}")
    return metadata_uri


def _load_keypair(private_key):
    """Load a Keypair from base58, hex, or JSON byte-array format."""
    from solders.keypair import Keypair

    stripped = private_key.strip()
    if stripped.startswith("["):
        raw = bytes(json.loads(stripped))
        return Keypair.from_bytes(raw) if len(raw) == 64 else Keypair.from_seed(raw)
    elif (all(c in "0123456789abcdefABCDEF" for c in stripped)
          and len(stripped) in (64, 128)):
        raw = bytes.fromhex(stripped)
        return Keypair.from_bytes(raw) if len(raw) == 64 else Keypair.from_seed(raw)
    else:
        return Keypair.from_base58_string(stripped)


def _create_and_broadcast(signer_private_key, rpc_url, metadata_uri,
                          name, symbol, initial_buy_sol, slippage_bps,
                          priority_fee):
    """Build, sign, and broadcast a pump.fun token-creation transaction."""
    import base58
    import requests
    from solders.keypair import Keypair
    from solders.transaction import VersionedTransaction

    mint_keypair = Keypair()
    mint_pubkey = str(mint_keypair.pubkey())
    signer_keypair = _load_keypair(signer_private_key)

    payload = {
        "publicKey": str(signer_keypair.pubkey()),
        "action": "create",
        "tokenMetadata": {"name": name, "symbol": symbol, "uri": metadata_uri},
        "mint": mint_pubkey,
        "denominatedInSol": "true",
        "amount": initial_buy_sol,
        "slippage": slippage_bps,
        "priorityFee": priority_fee,
        "pool": "pump",
    }

    resp = requests.post(PUMPPORTAL_API_URL, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"PumpPortal API error (HTTP {resp.status_code}): {resp.text}")
    if resp.content[:1] in (b"{", b"[", b'"'):
        raise RuntimeError(f"PumpPortal API error: {resp.text}")

    unsigned_tx = VersionedTransaction.from_bytes(resp.content)
    signed_tx = VersionedTransaction(unsigned_tx.message, [mint_keypair, signer_keypair])

    rpc_resp = requests.post(
        rpc_url,
        json={
            "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
            "params": [
                base58.b58encode(bytes(signed_tx)).decode("ascii"),
                {"skipPreflight": False, "preflightCommitment": "confirmed",
                 "encoding": "base58", "maxRetries": 3},
            ],
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

    rpc_result = rpc_resp.json()
    if "error" in rpc_result:
        raise RuntimeError(f"Solana RPC error: {rpc_result['error']}")

    return {"signature": rpc_result.get("result", ""), "mint": mint_pubkey}


# =====================================================================
# four.meme (BSC) helpers
# =====================================================================

def _fourmeme_api_post(endpoint, json_data=None, headers=None):
    """POST to four.meme API and return data."""
    import requests
    url = f"{FOUR_MEME_API}{endpoint}"
    resp = requests.post(url, json=json_data, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"four.meme API error (HTTP {resp.status_code}): {resp.text}")
    result = resp.json()
    if result.get("code") and result["code"] != "0":
        raise RuntimeError(f"four.meme API error: {result}")
    return result.get("data", result)


def _fourmeme_login(account_address, private_key):
    """Authenticate with four.meme and return access token."""
    from eth_account import Account
    from eth_account.messages import encode_defunct

    nonce = _fourmeme_api_post("/private/user/nonce/generate", {
        "accountAddress": account_address,
        "verifyType": "LOGIN",
        "networkCode": "BSC",
    })

    message = f"You are sign in Meme {nonce}"
    msg_hash = encode_defunct(text=message)
    signed = Account.sign_message(msg_hash, private_key)
    signature = signed.signature.hex()
    if not signature.startswith("0x"):
        signature = "0x" + signature

    access_token = _fourmeme_api_post("/private/user/login/dex", {
        "region": "WEB",
        "langType": "EN",
        "loginIp": "",
        "inviteCode": "",
        "verifyInfo": {
            "address": account_address,
            "networkCode": "BSC",
            "signature": signature,
            "verifyType": "LOGIN",
        },
        "walletName": "MetaMask",
    })
    return access_token


def _fourmeme_upload_image(image_path, access_token):
    """Upload token image to four.meme. Returns hosted image URL."""
    import requests

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg",
                  ".jpeg": "image/jpeg", ".gif": "image/gif",
                  ".bmp": "image/bmp", ".webp": "image/webp"}
    mime_type = mime_types.get(ext, "application/octet-stream")

    url = f"{FOUR_MEME_API}/private/token/upload"
    with open(image_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (os.path.basename(image_path), f, mime_type)},
            headers={"meme-web-access": access_token},
            timeout=30,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"four.meme image upload failed (HTTP {resp.status_code}): {resp.text}")
    result = resp.json()
    if result.get("code") and result["code"] != "0":
        raise RuntimeError(f"four.meme image upload failed: {result}")
    return result.get("data", result)


def _fourmeme_prepare_create(access_token, name, symbol, description,
                             image_url, label, presale_bnb=0,
                             twitter="", telegram="", website=""):
    """Call four.meme API to get createArg and signature for on-chain call."""
    import requests

    payload = {
        "name": name,
        "shortName": symbol,
        "desc": description,
        "imgUrl": image_url,
        "launchTime": int(time.time() * 1000) + 60000,
        "label": label,
        "preSale": str(presale_bnb),
        "onlyMPC": False,
        "lpTradingFee": 0.0025,
        # Fixed protocol parameters
        "totalSupply": 1000000000,
        "raisedAmount": 24,
        "saleRate": 0.8,
        "reserveRate": 0,
        "funGroup": False,
        "clickFun": False,
        "symbol": "BNB",
        "symbolAddress": "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c",
    }
    if twitter:
        payload["twitterUrl"] = twitter
    if telegram:
        payload["telegramUrl"] = telegram
    if website:
        payload["webUrl"] = website

    url = f"{FOUR_MEME_API}/private/token/create"
    resp = requests.post(url, json=payload,
                         headers={"meme-web-access": access_token}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"four.meme create API failed (HTTP {resp.status_code}): {resp.text}")
    result = resp.json()
    if result.get("code") and result["code"] != "0":
        raise RuntimeError(f"four.meme create API failed: {result}")

    data = result.get("data", result)
    create_arg = data.get("createArg") or data.get("create_arg") or data.get("arg")
    signature = data.get("signature") or data.get("sign") or data.get("signatureHex")
    if not create_arg or not signature:
        raise RuntimeError(f"Unexpected four.meme API response, keys: {list(data.keys())}")

    return create_arg, signature


def _fourmeme_create_onchain(w3, private_key, create_arg_hex, signature_hex,
                             presale_bnb=0):
    """Call TokenManager2.createToken on BSC and return tx receipt."""
    from eth_account import Account
    from web3 import Web3

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(TOKEN_MANAGER2),
        abi=TOKEN_MANAGER2_ABI,
    )
    account = Account.from_key(private_key)

    value_wei = Web3.to_wei(CREATE_FEE_BNB + presale_bnb, "ether")
    gwei = 10**9
    value_wei = (value_wei // gwei) * gwei

    create_arg_bytes = bytes.fromhex(create_arg_hex.replace("0x", ""))
    signature_bytes = bytes.fromhex(signature_hex.replace("0x", ""))

    tx = contract.functions.createToken(
        create_arg_bytes, signature_bytes
    ).build_transaction({
        "from": account.address,
        "value": value_wei,
        "gas": 3000000,
        "gasPrice": w3.eth.gas_price,
        "nonce": w3.eth.get_transaction_count(account.address),
        "chainId": w3.eth.chain_id,
    })

    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    return receipt


def _fourmeme_extract_token_address(w3, receipt):
    """Parse TokenCreate event from tx receipt to get the token address."""
    from web3 import Web3

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(TOKEN_MANAGER2),
        abi=TOKEN_MANAGER2_ABI,
    )
    logs = contract.events.TokenCreate().process_receipt(receipt)
    if not logs:
        raise RuntimeError("TokenCreate event not found in transaction receipt")
    event = logs[0]["args"]
    return {
        "token": event["token"],
        "name": event["name"],
        "symbol": event["symbol"],
        "requestId": event["requestId"],
    }


# =====================================================================
# Unified _run_create dispatcher
# =====================================================================

def _run_create_pumpfun(params: dict) -> dict:
    """Create token on pump.fun (Solana)."""
    private_key, rpc_url = _get_solana_credentials()
    missing = []
    if not private_key:
        missing.append("solanaPrivateKey / SOLANA_PRIVATE_KEY")
    if not rpc_url:
        missing.append("solanaRpcUrl / SOLANA_RPC_URL")
    if missing:
        return {"success": False,
                "error": f"Missing Solana credentials: {', '.join(missing)}. "
                         f"Set them in config.json under tools.memeMonitor or as environment variables."}

    metadata_uri = _upload_to_ipfs(
        image_path=params["image_path"],
        name=params["name"],
        symbol=params["symbol"],
        description=params["description"],
        twitter=params.get("twitter", ""),
        telegram=params.get("telegram", ""),
        website=params.get("website", ""),
    )

    result = _create_and_broadcast(
        signer_private_key=private_key,
        rpc_url=rpc_url,
        metadata_uri=metadata_uri,
        name=params["name"],
        symbol=params["symbol"],
        initial_buy_sol=float(params.get("buy_amount", DEFAULT_BUY_AMOUNT)),
        slippage_bps=int(params.get("slippage_bps", DEFAULT_SLIPPAGE_BPS)),
        priority_fee=float(params.get("priority_fee", DEFAULT_PRIORITY_FEE)),
    )

    mint, sig = result["mint"], result["signature"]
    return {
        "success": True, "platform": "pump.fun",
        "mint": mint, "signature": sig,
        "pump_fun_url": f"https://pump.fun/{mint}",
        "solscan_url": f"https://solscan.io/tx/{sig}",
    }


def _run_create_fourmeme(params: dict) -> dict:
    """Create token on four.meme (BSC)."""
    from eth_account import Account
    from web3 import Web3

    private_key, rpc_url = _get_bsc_credentials()
    if not private_key:
        return {"success": False,
                "error": "Missing BSC credentials: bscPrivateKey / BSC_PRIVATE_KEY. "
                         "Set it in config.json under tools.memeMonitor or as an environment variable."}

    label = params.get("label", "Meme")
    if label not in VALID_LABELS:
        return {"success": False,
                "error": f"Invalid label: '{label}'. Valid labels: {', '.join(VALID_LABELS)}"}

    presale_bnb = float(params.get("presale_bnb", DEFAULT_PRESALE_BNB))

    # Derive wallet address
    account = Account.from_key(private_key)
    wallet_address = account.address

    # Step 1: Login
    access_token = _fourmeme_login(wallet_address, private_key)

    # Step 2: Upload image
    image_url = _fourmeme_upload_image(params["image_path"], access_token)

    # Step 3: Prepare (get createArg + signature)
    create_arg, signature = _fourmeme_prepare_create(
        access_token=access_token,
        name=params["name"],
        symbol=params["symbol"],
        description=params["description"],
        image_url=image_url,
        label=label,
        presale_bnb=presale_bnb,
        twitter=params.get("twitter", ""),
        telegram=params.get("telegram", ""),
        website=params.get("website", ""),
    )

    # Step 4: On-chain transaction
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to BSC RPC: {rpc_url}")

    receipt = _fourmeme_create_onchain(w3, private_key, create_arg, signature, presale_bnb)

    if receipt["status"] != 1:
        raise RuntimeError("Transaction reverted on chain")

    token_info = _fourmeme_extract_token_address(w3, receipt)
    token_addr = token_info["token"]
    tx_hash = receipt["transactionHash"].hex()

    return {
        "success": True, "platform": "four.meme",
        "token_address": token_addr, "tx_hash": tx_hash,
        "four_meme_url": f"https://four.meme/token/{token_addr}",
        "bscscan_url": f"https://bscscan.com/tx/{tx_hash}",
    }


def _run_create(params: dict) -> dict:
    """Synchronous memecoin creation (runs in thread)."""
    platform = params.get("platform", "").lower().strip()

    if platform == "pump.fun":
        return _run_create_pumpfun(params)

    if platform in ("four.meme", "fourmeme", "bsc"):
        return _run_create_fourmeme(params)

    return {"success": False,
            "error": f"Unsupported platform: '{platform}'. Supported: pump.fun, four.meme"}


# =====================================================================
# Tool class
# =====================================================================

# Pending creation requests storage (maps confirmation_token → {params, created_at})
_pending_creates: dict[str, dict] = {}


class MemeCreateTool(Tool):
    """Tool to create (deploy) memecoins on pump.fun (Solana) or four.meme (BSC)."""

    @property
    def name(self) -> str:
        return "meme_create"

    @property
    def description(self) -> str:
        return (
            "Create and deploy a new memecoin on a blockchain. "
            "Supports pump.fun (Solana) and four.meme (BSC). "
            "Use 'check_env' command first to verify that wallet credentials are configured "
            "(in config.json under tools.memeMonitor, or as environment variables). "
            "If they are missing, tell the user to configure them before proceeding. "
            "Use 'create' command to deploy the token after env check passes. "
            "IMPORTANT: Always confirm token details with the user before creating."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["check_env", "create", "confirm", "cancel"],
                    "description": (
                        "check_env: verify wallet credentials are configured (config or env vars); "
                        "create: deploy the memecoin (requires credentials to be set); "
                        "confirm: confirm a pending token deployment using confirmation_token; "
                        "cancel: cancel a pending token deployment using confirmation_token."
                    ),
                },
                "confirmation_token": {
                    "type": "string",
                    "description": "Confirmation token for confirm/cancel commands. Required for 'confirm' and 'cancel'.",
                },
                "platform": {
                    "type": "string",
                    "enum": ["pump.fun", "four.meme"],
                    "description": "Target platform. pump.fun = Solana, four.meme = BSC. Required for 'create'.",
                },
                "name": {
                    "type": "string",
                    "description": "Token name. Required for 'create'.",
                },
                "symbol": {
                    "type": "string",
                    "description": "Token ticker symbol. Required for 'create'.",
                },
                "description": {
                    "type": "string",
                    "description": "Short token description. Required for 'create'.",
                },
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to logo image (PNG/JPG/GIF). Required for 'create'.",
                },
                "label": {
                    "type": "string",
                    "enum": VALID_LABELS,
                    "description": "Token category label. Required for four.meme, ignored for pump.fun. Default: 'Meme'.",
                },
                "buy_amount": {
                    "type": "number",
                    "description": "Initial buy amount in SOL (pump.fun only). Default: 0.01.",
                },
                "slippage_bps": {
                    "type": "integer",
                    "description": "Slippage in basis points (pump.fun only). Default: 10.",
                },
                "priority_fee": {
                    "type": "number",
                    "description": "Priority fee in SOL (pump.fun only). Default: 0.0005.",
                },
                "presale_bnb": {
                    "type": "number",
                    "description": "Creator presale amount in BNB (four.meme only). Default: 0.",
                },
                "twitter": {
                    "type": "string",
                    "description": "Twitter/X URL (optional).",
                },
                "telegram": {
                    "type": "string",
                    "description": "Telegram URL (optional).",
                },
                "website": {
                    "type": "string",
                    "description": "Website URL (optional).",
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        logger.info(f"meme_create:{command}")

        if command == "check_env":
            return self._check_env(kwargs.get("platform"))

        if command == "create":
            return await self._create(kwargs)

        if command == "confirm":
            return await self._confirm(kwargs)

        if command == "cancel":
            return await self._cancel(kwargs)

        return json.dumps({"error": f"Unknown command: {command!r}"})

    def _check_env(self, platform: str | None = None) -> str:
        results: dict[str, Any] = {}

        # Check Solana (pump.fun) if no platform specified or pump.fun requested
        if platform in (None, "pump.fun"):
            pk, rpc = _get_solana_credentials()
            sol_missing = []
            if not pk:
                sol_missing.append("solanaPrivateKey / SOLANA_PRIVATE_KEY")
            if not rpc:
                sol_missing.append("solanaRpcUrl / SOLANA_RPC_URL")
            results["pump.fun"] = {
                "ready": not sol_missing,
                **({"missing": sol_missing} if sol_missing else {}),
            }

        # Check BSC (four.meme) if no platform specified or four.meme/bsc requested
        if platform in (None, "four.meme", "fourmeme", "bsc"):
            pk, _ = _get_bsc_credentials()
            bsc_missing = []
            if not pk:
                bsc_missing.append("bscPrivateKey / BSC_PRIVATE_KEY")
            results["four.meme"] = {
                "ready": not bsc_missing,
                **({"missing": bsc_missing} if bsc_missing else {}),
            }

        all_ready = all(r.get("ready") for r in results.values())
        if not all_ready:
            return json.dumps({
                "ready": False,
                "platforms": results,
                "hint": "Set credentials in config.json under tools.memeMonitor or as environment variables.",
            })

        return json.dumps({"ready": True, "platforms": results,
                           "message": "Wallet credentials configured. Ready to create."})

    async def _create(self, kwargs: dict) -> str:
        import time as time_module

        global _pending_creates

        # Lazy cleanup of expired entries
        now = time_module.time()
        expired_tokens = [
            token for token, data in _pending_creates.items()
            if now - data.get("created_at", 0) > CONFIRMATION_TIMEOUT_SEC
        ]
        for token in expired_tokens:
            del _pending_creates[token]
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired creation requests")

        # Normalize field names (router uses token_name/token_description to
        # avoid collision with other router params)
        params = dict(kwargs)
        if "token_name" in params:
            params.setdefault("name", params.pop("token_name"))
        if "token_description" in params:
            params.setdefault("description", params.pop("token_description"))

        # Validate required fields
        required = ["platform", "name", "symbol", "description", "image_path"]
        missing = [f for f in required if not params.get(f)]
        if missing:
            return json.dumps({"success": False, "error": f"Missing required fields: {', '.join(missing)}"})

        # Generate confirmation token and store the request
        confirmation_token = str(uuid4())
        _pending_creates[confirmation_token] = {
            "params": params,
            "created_at": time_module.time(),
        }

        platform = params.get("platform", "").lower().strip()

        logger.info(f"Created confirmation token {confirmation_token} for token '{params.get('name')}' on {platform}")

        return json.dumps({
            "status": "pending_confirmation",
            "confirmation_token": confirmation_token,
            "details": {
                "name": params.get("name"),
                "symbol": params.get("symbol"),
                "platform": platform,
                "description": params.get("description"),
            },
            "message": "⚠️ Please confirm this token deployment. Use command='confirm' with this confirmation_token to proceed, or command='cancel' to abort. This expires in 5 minutes.",
        }, ensure_ascii=False)

    async def _confirm(self, kwargs: dict) -> str:
        import time as time_module

        global _pending_creates

        confirmation_token = kwargs.get("confirmation_token", "")
        if not confirmation_token:
            return json.dumps({"success": False, "error": "Missing required field: confirmation_token"})

        request_data = _pending_creates.get(confirmation_token)
        if not request_data:
            return json.dumps({"success": False, "error": "Invalid or expired confirmation token"})

        # Check if expired
        now = time_module.time()
        if now - request_data.get("created_at", 0) > CONFIRMATION_TIMEOUT_SEC:
            del _pending_creates[confirmation_token]
            return json.dumps({"success": False, "error": "Confirmation token has expired. Please create a new deployment request."})

        # Execute the creation
        params = request_data["params"]
        try:
            result = await asyncio.to_thread(_run_create, params)
            # Remove from pending on success (or failure - user can retry)
            del _pending_creates[confirmation_token]
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"meme_create confirm error: {e}")
            # Keep the pending request so user can retry if desired
            return json.dumps({"success": False, "error": str(e)})

    async def _cancel(self, kwargs: dict) -> str:
        global _pending_creates

        confirmation_token = kwargs.get("confirmation_token", "")
        if not confirmation_token:
            return json.dumps({"success": False, "error": "Missing required field: confirmation_token"})

        if confirmation_token not in _pending_creates:
            return json.dumps({"success": False, "error": "Invalid or expired confirmation token"})

        del _pending_creates[confirmation_token]
        logger.info(f"Cancelled creation request with token {confirmation_token}")

        return json.dumps({
            "success": True,
            "message": "Token deployment cancelled successfully.",
        })
