# Project Sovereign — PL Genesis Market Agent

## What This Is

Autonomous market intelligence agent submitted to the PL Genesis hackathon.
Democratizes institutional-grade investment research (~$0.05/analysis vs $24k/yr Bloomberg).
Runs a full discover -> plan -> execute -> verify decision loop with zero human intervention.
Every decision is logged to IPFS (Storacha) and linked to an on-chain identity (ERC-8004).

## Team Context

- **CC (Coalition Code)** — AI architect who designed and directed the build via Project Agent Army
- **Thomas (human operator)** — Strategy, coordination, and domain expertise
- **Org:** Liberation Labs / Transparent Humboldt Coalition

## Key Modules

| Directory | Purpose |
|-----------|---------|
| `core/` | ReAct agent (`react_agent.py`), tool registry, decision loop, narrator, investor profile |
| `execution/` | Order execution (Alpaca), portfolio management, risk manager, scanner, strategy |
| `safety/` | 8-layer guardrails, anomaly detection (price/volume/drift), pre-trade validation |
| `integrations/` | ERC-8004 (Base L2 identity), Storacha (IPFS audit logs), Lit Protocol (encrypted signals) |
| `analysis/` | Technical indicators (SMA, RSI, MACD, Bollinger), sentiment, congressional STOCK Act, FRED macro |
| `dashboard/` | Web dashboard for monitoring |
| `memory/` | Agent memory and knowledge graph |
| `docs/` | Architecture, safety philosophy, compliance, sponsor integration docs |

## How to Run

```bash
cp .env.example .env        # Add API keys (Alpaca, Anthropic, etc.)
make setup                   # Creates venv, installs deps
source .venv/bin/activate

python main.py --autonomous  # Full autonomous decision loop
python main.py --scan        # Market scan only
python main.py --query "..."  # Single analysis query
make dashboard               # Web dashboard
make monitor                 # System monitor
```

## Hackathon Tracks

Fresh Code, AI & Robotics, Agent Only, Agents With Receipts (ERC-8004),
Crypto (token-gated signals), Storacha (IPFS audit trail), Lit Protocol (encrypted access).

## Architecture Notes

- **ReAct pattern** — Reasoning + Acting agent powered by Claude with extensible tool registry
- **8-layer risk system** — Position sizing, macro overlay, sector limits, VIX-adaptive stops, circuit breakers
- **Narrator pattern** — Plain language explanations of all decisions
- **Immutable audit trail** — Structured JSON -> IPFS/Filecoin via Storacha
- Entry point: `main.py` | Config: `core/config.py` | Agent def: `agent.json`
