"""alpaca_monitor tool — READ-ONLY Alpaca paper trading position monitor.

Provides two tools for Christopher (Trading MD):
- alpaca_get_positions: list all open positions with current P&L
- alpaca_check_exit: evaluate positions against stop-loss/take-profit thresholds

All trade EXECUTION goes through the Alpaca MCP server (place_stock_order)
with HITL approval via the christopher-approval skill. This tool is
strictly read-only — it cannot place, modify, or close orders.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_ALPACA_REQUIRED_ENV = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]


def _check_alpaca_env() -> bool:
    return all(os.getenv(k) for k in _ALPACA_REQUIRED_ENV)


def _get_trading_client():
    from alpaca.trading.client import TradingClient
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1", "yes")
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)


def alpaca_get_positions() -> Dict[str, Any]:
    try:
        client = _get_trading_client()
        positions = client.get_all_positions()
        if not positions:
            return {"positions": [], "count": 0, "summary": "No open positions"}

        result = []
        total_unrealized_pl = 0.0
        total_market_value = 0.0
        for p in positions:
            entry = {
                "symbol": p.symbol,
                "qty": str(p.qty),
                "side": p.side,
                "market_value": float(p.market_value) if p.market_value else 0.0,
                "cost_basis": float(p.cost_basis) if p.cost_basis else 0.0,
                "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else 0.0,
                "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else 0.0,
                "current_price": float(p.current_price) if p.current_price else 0.0,
                "avg_entry_price": float(p.avg_entry_price) if p.avg_entry_price else 0.0,
                "change_today": float(p.change_today) if p.change_today else 0.0,
            }
            result.append(entry)
            total_unrealized_pl += entry["unrealized_pl"]
            total_market_value += entry["market_value"]

        total_plpc = (total_unrealized_pl / total_market_value * 100) if total_market_value else 0.0
        return {
            "positions": result,
            "count": len(result),
            "total_unrealized_pl": round(total_unrealized_pl, 2),
            "total_market_value": round(total_market_value, 2),
            "total_plpc": f"{total_plpc:.2f}%",
        }
    except Exception as e:
        logger.error("alpaca_get_positions error: %s", e)
        return tool_error(f"Failed to fetch positions: {e}")


def alpaca_check_exit(
    stop_loss_pct: float = -5.0,
    take_profit_pct: float = 15.0,
) -> Dict[str, Any]:
    try:
        client = _get_trading_client()
        positions = client.get_all_positions()
        if not positions:
            return {"alerts": [], "summary": "No open positions to check"}

        alerts = []
        for p in positions:
            plpc = float(p.unrealized_plpc) if p.unrealized_plpc else 0.0
            pct = plpc * 100
            if pct <= stop_loss_pct:
                alerts.append({
                    "symbol": p.symbol,
                    "alert": "STOP_LOSS",
                    "current_plpc": f"{pct:.2f}%",
                    "threshold": f"{stop_loss_pct:.2f}%",
                    "qty": str(p.qty),
                    "current_price": float(p.current_price) if p.current_price else 0.0,
                })
            elif pct >= take_profit_pct:
                alerts.append({
                    "symbol": p.symbol,
                    "alert": "TAKE_PROFIT",
                    "current_plpc": f"{pct:.2f}%",
                    "threshold": f"{take_profit_pct:.2f}%",
                    "qty": str(p.qty),
                    "current_price": float(p.current_price) if p.current_price else 0.0,
                })

        summary = f"{len(alerts)} alert(s)" if alerts else "No positions triggered stop-loss or take-profit thresholds"
        return {"alerts": alerts, "stop_loss_pct": f"{stop_loss_pct:.2f}%", "take_profit_pct": f"{take_profit_pct:.2f}%", "summary": summary}
    except Exception as e:
        logger.error("alpaca_check_exit error: %s", e)
        return tool_error(f"Failed to check exit conditions: {e}")


def alpaca_get_account() -> Dict[str, Any]:
    try:
        client = _get_trading_client()
        acct = client.get_account()
        return {
            "equity": float(acct.equity) if acct.equity else 0.0,
            "cash": float(acct.cash) if acct.cash else 0.0,
            "buying_power": float(acct.buying_power) if acct.buying_power else 0.0,
            "portfolio_value": float(acct.portfolio_value) if acct.portfolio_value else 0.0,
            "long_market_value": float(acct.long_market_value) if acct.long_market_value else 0.0,
            "short_market_value": float(acct.short_market_value) if acct.short_market_value else 0.0,
            "status": acct.status,
            "paper": True,
        }
    except Exception as e:
        logger.error("alpaca_get_account error: %s", e)
        return tool_error(f"Failed to fetch account: {e}")


ALPACA_GET_POSITIONS_SCHEMA = {
    "name": "alpaca_get_positions",
    "description": (
        "List all open positions in the Alpaca paper trading account with "
        "current P&L, market value, and per-position breakdown. READ-ONLY — "
        "does not place or modify orders. Use the Alpaca MCP tools "
        "(place_stock_order) for trade execution with HITL approval."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

ALPACA_CHECK_EXIT_SCHEMA = {
    "name": "alpaca_check_exit",
    "description": (
        "Evaluate open positions against stop-loss and take-profit thresholds. "
        "Returns alerts for positions breaching either threshold. READ-ONLY — "
        "does not close positions. Use the Alpaca MCP tools (place_stock_order) "
        "for trade execution with HITL approval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "stop_loss_pct": {
                "type": "number",
                "description": "Stop-loss threshold as negative percent (default -5.0 means exit if P&L drops 5%)",
                "default": -5.0,
            },
            "take_profit_pct": {
                "type": "number",
                "description": "Take-profit threshold as positive percent (default 15.0 means exit if P&L rises 15%)",
                "default": 15.0,
            },
        },
        "required": [],
    },
}

ALPACA_GET_ACCOUNT_SCHEMA = {
    "name": "alpaca_get_account",
    "description": (
        "Get Alpaca paper trading account summary: equity, cash, buying power, "
        "portfolio value. READ-ONLY."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

registry.register(
    name="alpaca_get_positions",
    toolset="alpaca",
    schema=ALPACA_GET_POSITIONS_SCHEMA,
    handler=lambda args, **kw: alpaca_get_positions(),
    check_fn=_check_alpaca_env,
    emoji="📊",
)

registry.register(
    name="alpaca_check_exit",
    toolset="alpaca",
    schema=ALPACA_CHECK_EXIT_SCHEMA,
    handler=lambda args, **kw: alpaca_check_exit(
        stop_loss_pct=args.get("stop_loss_pct", -5.0),
        take_profit_pct=args.get("take_profit_pct", 15.0),
    ),
    check_fn=_check_alpaca_env,
    emoji="🚨",
)

registry.register(
    name="alpaca_get_account",
    toolset="alpaca",
    schema=ALPACA_GET_ACCOUNT_SCHEMA,
    handler=lambda args, **kw: alpaca_get_account(),
    check_fn=_check_alpaca_env,
    emoji="💰",
)
