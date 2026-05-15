"""prediction_market_tool — READ-ONLY prediction market data tools for Christopher.

8 native Hermes tools under the 'prediction_markets' toolset:
- pm_search_markets: Search events/markets by keyword
- pm_get_market: Single market details (prices, volume, OI)
- pm_get_orderbook: Full depth order book
- pm_get_candlesticks: OHLCV price history
- pm_get_price_history: Historical close price series
- pm_get_trades: Recent executed trades
- pm_cross_market_signals: Cross-venue price comparison + ARB detection
- pm_volume_spike_scan: Unusual volume detection via z-score

All tools hit live APIs directly. No Qdrant cache. Accept 'platform' param
(kalshi/polymarket/both). All are READ-ONLY — no order placement.

Polymarket APIs (all public, no auth):
- Gamma: gamma-api.polymarket.com (events, search, tags)
- CLOB:  clob.polymarket.com (orderbook, prices, spreads, history)
- Data:  data-api.polymarket.com (trades, positions, OI)

Kalshi APIs:
- Public: api.elections.kalshi.com/trade-api/v2 (no auth)
- Demo:   demo-api.kalshi.co/trade-api/v2 (RSA-PSS auth)
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

KALSHI_PUBLIC_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLY_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLY_CLOB_BASE = "https://clob.polymarket.com"
POLY_DATA_BASE = "https://data-api.polymarket.com"

_http = httpx.Client(timeout=15.0)


def _check_pm_requirements() -> bool:
    try:
        import pykalshi  # noqa: F401
        return True
    except ImportError:
        return False


def _get_kalshi_client():
    from pykalshi import KalshiClient
    key_id = os.getenv("KALSHI_API_KEY", "")
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    demo = os.getenv("KALSHI_DEMO", "true").lower() in ("true", "1", "yes")
    return KalshiClient(api_key_id=key_id, private_key_path=key_path, demo=demo)


def _kalshi_public_get(path: str, params: Optional[dict] = None) -> Any:
    try:
        r = _http.get(f"{KALSHI_PUBLIC_BASE}{path}", params=params or {})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Kalshi public GET %s failed: %s", path, e)
        return None


def _poly_gamma_get(path: str, params: Optional[dict] = None) -> Any:
    try:
        r = _http.get(f"{POLY_GAMMA_BASE}{path}", params=params or {})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Poly Gamma GET %s failed: %s", path, e)
        return None


def _poly_clob_get(path: str, params: Optional[dict] = None) -> Any:
    try:
        r = _http.get(f"{POLY_CLOB_BASE}{path}", params=params or {})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Poly CLOB GET %s failed: %s", path, e)
        return None


def _poly_data_get(path: str, params: Optional[dict] = None) -> Any:
    try:
        r = _http.get(f"{POLY_DATA_BASE}{path}", params=params or {})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Poly Data GET %s failed: %s", path, e)
        return None


def _normalize_platform(platform: str) -> List[str]:
    if platform == "both":
        return ["kalshi", "polymarket"]
    return [platform]


# ---------------------------------------------------------------------------
# Tool 1: pm_search_markets
# ---------------------------------------------------------------------------

def pm_search_markets(query: str, platform: str = "both", limit: int = 10) -> Dict[str, Any]:
    results = []
    platforms = _normalize_platform(platform)

    if "kalshi" in platforms:
        try:
            c = _get_kalshi_client()
            events = c.get_events(limit=limit)
            q = query.lower()
            for e in events:
                title = getattr(e, "title", "") or ""
                if q in title.lower() or not query:
                    results.append({
                        "event_title": title,
                        "platform": "kalshi",
                        "markets_count": getattr(e, "markets_count", 0),
                        "category": getattr(e, "series_slug", ""),
                        "resolution_date": str(getattr(e, "expiration_date", "")),
                        "ticker": getattr(e, "event_ticker", ""),
                        "condition_id": None,
                    })
        except Exception as e:
            logger.error("Kalshi search failed: %s", e)

    if "polymarket" in platforms:
        data = _poly_gamma_get("/events", {"limit": limit, "active": "true"})
        if data:
            items = data if isinstance(data, list) else data.get("results", data.get("data", []))
            q = query.lower()
            for ev in items[:limit]:
                title = ev.get("title", "") or ""
                if q in title.lower() or not query:
                    results.append({
                        "event_title": title,
                        "platform": "polymarket",
                        "markets_count": len(ev.get("markets", [])),
                        "category": ev.get("slug", ""),
                        "resolution_date": ev.get("end_date_iso", ""),
                        "ticker": None,
                        "condition_id": ev.get("condition_id", ""),
                    })

    return {"results": results[:limit], "count": len(results), "platforms_queried": platforms}


# ---------------------------------------------------------------------------
# Tool 2: pm_get_market
# ---------------------------------------------------------------------------

def pm_get_market(ticker: str = "", condition_id: str = "", platform: str = "kalshi") -> Dict[str, Any]:
    if platform == "kalshi" and ticker:
        try:
            c = _get_kalshi_client()
            m = c.get_market(ticker)
            return {
                "ticker": ticker,
                "platform": "kalshi",
                "title": getattr(m, "title", ""),
                "yes_price": float(getattr(m, "yes_price", 0) or 0) / 100,
                "no_price": float(getattr(m, "no_price", 0) or 0) / 100,
                "volume": int(getattr(m, "volume", 0) or 0),
                "open_interest": int(getattr(m, "open_interest", 0) or 0),
                "spread": None,
                "best_bid": float(getattr(m, "yes_bid", 0) or 0) / 100,
                "best_ask": float(getattr(m, "yes_ask", 0) or 0) / 100,
                "resolution_date": str(getattr(m, "close_time", "")),
                "rules": getattr(m, "resolution_source", ""),
                "category": getattr(m, "category", ""),
                "status": getattr(m, "status", ""),
            }
        except Exception as e:
            return tool_error(f"Kalshi get_market failed: {e}")

    if platform == "polymarket":
        token_id = condition_id or ticker
        market = _poly_clob_get(f"/markets/{token_id}") if token_id else None
        if not market:
            search = _poly_gamma_get("/markets", {"limit": 1})
            if search:
                items = search if isinstance(search, list) else search.get("results", search.get("data", []))
                if items:
                    market = items[0]
        if market:
            price_data = _poly_clob_get("/midpoint", {"token_id": token_id}) if token_id else None
            spread_data = _poly_clob_get("/spread", {"token_id": token_id}) if token_id else None
            mid = float(price_data.get("mid", 0.5)) if price_data else None
            spread = float(spread_data.get("spread", 0)) if spread_data else None
            return {
                "ticker": token_id,
                "platform": "polymarket",
                "title": market.get("question", market.get("title", "")),
                "yes_price": mid,
                "no_price": round(1 - mid, 4) if mid is not None else None,
                "volume": market.get("volume", 0),
                "open_interest": market.get("open_interest", 0),
                "spread": spread,
                "best_bid": None,
                "best_ask": None,
                "resolution_date": market.get("end_date_iso", ""),
                "rules": market.get("description", ""),
                "category": market.get("category", ""),
                "status": market.get("active", ""),
            }
        return tool_error("Polymarket market not found")

    return tool_error("Specify platform=kalshi with ticker, or platform=polymarket with condition_id")


# ---------------------------------------------------------------------------
# Tool 3: pm_get_orderbook
# ---------------------------------------------------------------------------

def pm_get_orderbook(ticker: str = "", token_id: str = "", platform: str = "kalshi", depth: int = 20) -> Dict[str, Any]:
    if platform == "kalshi" and ticker:
        data = _kalshi_public_get(f"/markets/{ticker}/orderbook", {"depth": depth})
        if data:
            bids = [{"price": float(b.get("price", 0)) / 100, "size": int(b.get("size", 0))} for b in data.get("bids", [])[:depth]]
            asks = [{"price": float(a.get("price", 0)) / 100, "size": int(a.get("size", 0))} for a in data.get("asks", [])[:depth]]
            best_bid = bids[0]["price"] if bids else None
            best_ask = asks[0]["price"] if asks else None
            return {
                "ticker": ticker, "platform": "kalshi", "bids": bids, "asks": asks,
                "best_bid": best_bid, "best_ask": best_ask,
                "spread": round(best_ask - best_bid, 4) if best_bid and best_ask else None,
                "midpoint": round((best_bid + best_ask) / 2, 4) if best_bid and best_ask else None,
                "depth": depth,
            }
        return tool_error(f"Kalshi orderbook failed for {ticker}")

    if platform == "polymarket" and token_id:
        data = _poly_clob_get("/order-book", {"token_id": token_id})
        if data:
            bids = [{"price": float(b.get("price", 0)), "size": float(b.get("size", 0))} for b in data.get("bids", [])[:depth]]
            asks = [{"price": float(a.get("price", 0)), "size": float(a.get("size", 0))} for a in data.get("asks", [])[:depth]]
            best_bid = bids[0]["price"] if bids else None
            best_ask = asks[0]["price"] if asks else None
            return {
                "ticker": token_id, "platform": "polymarket", "bids": bids, "asks": asks,
                "best_bid": best_bid, "best_ask": best_ask,
                "spread": round(best_ask - best_bid, 4) if best_bid and best_ask else None,
                "midpoint": round((best_bid + best_ask) / 2, 4) if best_bid and best_ask else None,
                "depth": depth,
            }
        return tool_error(f"Polymarket orderbook failed for {token_id}")

    return tool_error("Specify platform with ticker (kalshi) or token_id (polymarket)")


# ---------------------------------------------------------------------------
# Tool 4: pm_get_candlesticks
# ---------------------------------------------------------------------------

def pm_get_candlesticks(ticker: str = "", token_id: str = "", platform: str = "kalshi",
                        interval: str = "1day", count: int = 30) -> Dict[str, Any]:
    if platform == "kalshi" and ticker:
        try:
            c = _get_kalshi_client()
            market = c.get_market(ticker)
            sticks = market.get_candlesticks(interval=interval, count=count)
            candles = []
            for s in sticks:
                candles.append({
                    "timestamp": str(getattr(s, "start_time", "")),
                    "open": float(getattr(s, "open", 0) or 0) / 100,
                    "high": float(getattr(s, "high", 0) or 0) / 100,
                    "low": float(getattr(s, "low", 0) or 0) / 100,
                    "close": float(getattr(s, "close", 0) or 0) / 100,
                    "volume": int(getattr(s, "volume", 0) or 0),
                })
            return {"ticker": ticker, "platform": "kalshi", "interval": interval, "candles": candles}
        except Exception as e:
            return tool_error(f"Kalshi candlesticks failed: {e}")

    if platform == "polymarket" and token_id:
        data = _poly_clob_get("/prices-history", {"market": token_id, "interval": interval, "n": count})
        if data:
            candles = []
            for p in data.get("history", [])[:count]:
                candles.append({
                    "timestamp": str(p.get("t", "")),
                    "open": None, "high": None, "low": None,
                    "close": float(p.get("p", 0)),
                    "volume": None,
                })
            return {"ticker": token_id, "platform": "polymarket", "interval": interval, "candles": candles}
        return tool_error(f"Polymarket price history failed for {token_id}")

    return tool_error("Specify platform with ticker (kalshi) or token_id (polymarket)")


# ---------------------------------------------------------------------------
# Tool 5: pm_get_price_history
# ---------------------------------------------------------------------------

def pm_get_price_history(ticker: str = "", token_id: str = "", platform: str = "kalshi",
                         days: int = 30) -> Dict[str, Any]:
    result = pm_get_candlesticks(ticker=ticker, token_id=token_id, platform=platform,
                                 interval="1day", count=days)
    if "error" in result:
        return result
    prices = [{"date": c["timestamp"], "price": c["close"]} for c in result.get("candles", []) if c.get("close") is not None]
    return {"ticker": result.get("ticker", ""), "platform": platform, "prices": prices, "count": len(prices)}


# ---------------------------------------------------------------------------
# Tool 6: pm_get_trades
# ---------------------------------------------------------------------------

def pm_get_trades(ticker: str = "", token_id: str = "", platform: str = "kalshi",
                  limit: int = 20) -> Dict[str, Any]:
    if platform == "kalshi" and ticker:
        data = _kalshi_public_get(f"/markets/{ticker}/trades", {"limit": limit})
        if data:
            trades = []
            for t in data.get("trades", [])[:limit]:
                trades.append({
                    "price": float(t.get("price", 0)) / 100,
                    "size": int(t.get("size", 0)),
                    "side": t.get("side", ""),
                    "timestamp": t.get("created_time", ""),
                    "platform": "kalshi",
                })
            return {"ticker": ticker, "platform": "kalshi", "trades": trades, "count": len(trades)}
        return tool_error(f"Kalshi trades failed for {ticker}")

    if platform == "polymarket" and token_id:
        data = _poly_data_get("/trade", {"asset_slug": token_id, "limit": limit})
        if data:
            trades = []
            items = data if isinstance(data, list) else data.get("trades", data.get("results", []))
            for t in (items or [])[:limit]:
                trades.append({
                    "price": float(t.get("price", 0)),
                    "size": float(t.get("size", 0)),
                    "side": t.get("side", ""),
                    "timestamp": t.get("timestamp", ""),
                    "platform": "polymarket",
                })
            return {"ticker": token_id, "platform": "polymarket", "trades": trades, "count": len(trades)}
        return tool_error(f"Polymarket trades failed for {token_id}")

    return tool_error("Specify platform with ticker (kalshi) or token_id (polymarket)")


# ---------------------------------------------------------------------------
# Tool 7: pm_cross_market_signals
# ---------------------------------------------------------------------------

def pm_cross_market_signals(query: str, min_gap_pct: float = 1.0, limit: int = 10) -> Dict[str, Any]:
    signals = []
    correlated = []

    kalshi_events = []
    poly_events = []

    try:
        c = _get_kalshi_client()
        kalshi_events = c.get_events(limit=limit * 3)
    except Exception as e:
        logger.error("Cross-signal Kalshi failed: %s", e)

    poly_data = _poly_gamma_get("/events", {"limit": limit * 3, "active": "true"})
    if poly_data:
        poly_events = poly_data if isinstance(poly_data, list) else poly_data.get("results", poly_data.get("data", []))

    q = query.lower()
    kalshi_matched = [e for e in kalshi_events if q in (getattr(e, "title", "") or "").lower()]
    poly_matched = [e for e in poly_events if q in (e.get("title", "") or "").lower()]

    for ke in kalshi_matched:
        k_title = getattr(ke, "title", "").lower()
        k_ticker = getattr(ke, "event_ticker", "")
        k_yes = float(getattr(ke, "yes_price", 0) or 0) / 100

        for pe in poly_matched:
            p_title = (pe.get("title", "") or "").lower()
            overlap = sum(1 for w in k_title.split() if w in p_title and len(w) > 3)
            if overlap < 2:
                continue

            p_markets = pe.get("markets", [])
            p_yes = None
            p_cond = None
            if p_markets:
                p_cond = p_markets[0].get("condition_id", "")
                p_price = _poly_clob_get("/midpoint", {"token_id": p_cond}) if p_cond else None
                p_yes = float(p_price.get("mid", 0.5)) if p_price else None

            if p_yes is not None and k_yes is not None:
                gap = abs(k_yes - p_yes)
                gap_pct = gap * 100
                arb = gap_pct >= min_gap_pct

                if k_yes + p_yes < 0.98:
                    arb = True

                signals.append({
                    "kalshi_ticker": k_ticker,
                    "kalshi_yes_price": k_yes,
                    "polymarket_condition_id": p_cond,
                    "polymarket_yes_price": p_yes,
                    "price_gap": round(gap, 4),
                    "arb_opportunity": arb,
                    "arb_type": "cross_venue" if arb else None,
                    "arb_return_pct": round(gap_pct, 2),
                    "kalshi_volume": getattr(ke, "volume", 0),
                    "polymarket_volume": pe.get("volume", 0),
                    "volume_ratio": None,
                    "resolution_date_match": None,
                })

    kalshi_by_cat = {}
    for e in kalshi_matched:
        cat = getattr(e, "series_slug", "unknown")
        kalshi_by_cat.setdefault(cat, []).append(e)

    for cat, events in kalshi_by_cat.items():
        if len(events) >= 2:
            correlated.append({
                "group": cat,
                "markets": [{"ticker": getattr(e, "event_ticker", ""), "title": getattr(e, "title", "")} for e in events[:5]],
                "correlation": "positive",
                "note": "Same series — adjacent outcomes likely positively correlated",
            })

    arb_count = sum(1 for s in signals if s.get("arb_opportunity"))
    summary = f"{arb_count} ARB gap(s) found, {len(correlated)} correlated group(s)"
    if not signals:
        summary = f"No cross-market signals for '{query}'"

    return {"signals": signals[:limit], "correlated_markets": correlated, "summary": summary}


# ---------------------------------------------------------------------------
# Tool 8: pm_volume_spike_scan
# ---------------------------------------------------------------------------

def pm_volume_spike_scan(threshold_z: float = 2.0, lookback_days: int = 7,
                         limit: int = 10) -> Dict[str, Any]:
    alerts = []
    k_count = 0
    p_count = 0

    try:
        c = _get_kalshi_client()
        events = c.get_events(limit=50)
        for e in events:
            try:
                ticker = getattr(e, "event_ticker", "")
                if not ticker:
                    continue
                market = c.get_market(ticker)
                cur_vol = int(getattr(market, "volume", 0) or 0)
                sticks = market.get_candlesticks(interval="1day", count=lookback_days)
                if not sticks:
                    continue
                vols = [int(getattr(s, "volume", 0) or 0) for s in sticks]
                avg_vol = sum(vols) / len(vols) if vols else 1
                if avg_vol < 1:
                    continue
                z = (cur_vol - avg_vol) / (math.sqrt(sum((v - avg_vol) ** 2 for v in vols) / len(vols)) if len(vols) > 1 else avg_vol)
                k_count += 1
                if abs(z) >= threshold_z:
                    alerts.append({
                        "ticker": ticker,
                        "platform": "kalshi",
                        "current_volume": cur_vol,
                        "avg_volume": int(avg_vol),
                        "z_score": round(z, 2),
                        "yes_price": float(getattr(market, "yes_price", 0) or 0) / 100,
                        "price_change_24h": None,
                        "alert": "VOLUME_SPIKE",
                    })
            except Exception:
                k_count += 1
                continue
    except Exception as e:
        logger.error("Kalshi volume scan failed: %s", e)

    poly_markets = _poly_gamma_get("/markets", {"limit": 50, "active": "true", "order": "volume24hr", "ascending": "false"})
    if poly_markets:
        items = poly_markets if isinstance(poly_markets, list) else poly_markets.get("results", poly_markets.get("data", []))
        for m in items:
            p_count += 1
            vol_24h = float(m.get("volume24hr", 0) or 0)
            vol_total = float(m.get("volume", 0) or 0)
            if vol_total < 1 or vol_24h < 1:
                continue
            cond_id = m.get("condition_id", "")
            hist = _poly_clob_get("/prices-history", {"market": cond_id, "interval": "1day", "n": lookback_days}) if cond_id else None
            if hist and hist.get("history"):
                vols = [float(p.get("p", 0)) for p in hist["history"]]
                avg_vol = sum(vols) / len(vols) if vols else 1
                if avg_vol < 0.001:
                    continue
                z = (vol_24h - avg_vol) / (math.sqrt(sum((v - avg_vol) ** 2 for v in vols) / len(vols)) if len(vols) > 1 else 1)
                if abs(z) >= threshold_z:
                    alerts.append({
                        "ticker": cond_id,
                        "platform": "polymarket",
                        "current_volume": vol_24h,
                        "avg_volume": round(avg_vol, 2),
                        "z_score": round(z, 2),
                        "yes_price": None,
                        "price_change_24h": None,
                        "alert": "VOLUME_SPIKE",
                    })

    alerts.sort(key=lambda a: abs(a.get("z_score", 0)), reverse=True)
    return {
        "alerts": alerts[:limit],
        "markets_scanned": {"kalshi": k_count, "polymarket": p_count},
        "threshold_z": threshold_z,
        "summary": f"{len(alerts)} volume spike(s) detected across {k_count} Kalshi + {p_count} Poly markets",
    }


# ---------------------------------------------------------------------------
# Schemas + Registration
# ---------------------------------------------------------------------------

PM_SEARCH_MARKETS_SCHEMA = {
    "name": "pm_search_markets",
    "description": (
        "Search prediction market events and markets by keyword across Kalshi and/or Polymarket. "
        "Returns matching events with title, platform, category, and resolution date. "
        "READ-ONLY — no trading."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keyword (e.g. 'Fed rate', 'Ukraine', 'Bitcoin')"},
            "platform": {"type": "string", "enum": ["kalshi", "polymarket", "both"], "default": "both", "description": "Which platform(s) to search"},
            "limit": {"type": "integer", "description": "Max results per platform", "default": 10},
        },
        "required": ["query"],
    },
}

PM_GET_MARKET_SCHEMA = {
    "name": "pm_get_market",
    "description": (
        "Get single prediction market details: YES/NO prices, volume, open interest, "
        "spread, resolution date, and rules. Use Kalshi ticker or Polymarket condition_id. "
        "READ-ONLY."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Kalshi market ticker (e.g. 'KX100FEDFUND-JUN26')"},
            "condition_id": {"type": "string", "description": "Polymarket condition ID"},
            "platform": {"type": "string", "enum": ["kalshi", "polymarket"], "default": "kalshi"},
        },
        "required": [],
    },
}

PM_GET_ORDERBOOK_SCHEMA = {
    "name": "pm_get_orderbook",
    "description": (
        "Get full depth order book for a prediction market. Shows bids, asks, best bid/ask, "
        "spread, and midpoint. READ-ONLY."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Kalshi market ticker"},
            "token_id": {"type": "string", "description": "Polymarket token ID"},
            "platform": {"type": "string", "enum": ["kalshi", "polymarket"], "default": "kalshi"},
            "depth": {"type": "integer", "description": "Order book depth (default 20)", "default": 20},
        },
        "required": [],
    },
}

PM_GET_CANDLESTICKS_SCHEMA = {
    "name": "pm_get_candlesticks",
    "description": (
        "Get OHLCV candlestick data for a prediction market. Intervals: 1min, 1hour, 1day. "
        "READ-ONLY."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Kalshi market ticker"},
            "token_id": {"type": "string", "description": "Polymarket token/condition ID"},
            "platform": {"type": "string", "enum": ["kalshi", "polymarket"], "default": "kalshi"},
            "interval": {"type": "string", "enum": ["1minute", "1hour", "1day"], "default": "1day", "description": "Candlestick interval"},
            "count": {"type": "integer", "description": "Number of candles (default 30)", "default": 30},
        },
        "required": [],
    },
}

PM_GET_PRICE_HISTORY_SCHEMA = {
    "name": "pm_get_price_history",
    "description": (
        "Get historical daily close price series for a prediction market. "
        "Simpler than candlesticks — just date + close price. READ-ONLY."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Kalshi market ticker"},
            "token_id": {"type": "string", "description": "Polymarket token/condition ID"},
            "platform": {"type": "string", "enum": ["kalshi", "polymarket"], "default": "kalshi"},
            "days": {"type": "integer", "description": "Number of days of history (default 30)", "default": 30},
        },
        "required": [],
    },
}

PM_GET_TRADES_SCHEMA = {
    "name": "pm_get_trades",
    "description": (
        "Get recent executed trades for a prediction market. Returns price, size, side, "
        "and timestamp. READ-ONLY."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Kalshi market ticker"},
            "token_id": {"type": "string", "description": "Polymarket token ID"},
            "platform": {"type": "string", "enum": ["kalshi", "polymarket"], "default": "kalshi"},
            "limit": {"type": "integer", "description": "Max trades to return (default 20)", "default": 20},
        },
        "required": [],
    },
}

PM_CROSS_MARKET_SIGNALS_SCHEMA = {
    "name": "pm_cross_market_signals",
    "description": (
        "Compare same-event prices across Kalshi and Polymarket. Detects ARB gaps "
        "(where YES_Kalshi + YES_Poly < 1.0 or price gap exceeds threshold), "
        "flags correlated contracts (same series, adjacent outcomes), and reports "
        "volume ratios. READ-ONLY."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Event keyword to search across both platforms"},
            "min_gap_pct": {"type": "number", "description": "Minimum price gap % to flag as ARB (default 1.0%)", "default": 1.0},
            "limit": {"type": "integer", "description": "Max signal pairs to return", "default": 10},
        },
        "required": ["query"],
    },
}

PM_VOLUME_SPIKE_SCAN_SCHEMA = {
    "name": "pm_volume_spike_scan",
    "description": (
        "Scan all active prediction markets for unusual volume spikes. Compares current "
        "volume against N-day average using z-score. Flags markets with |z| >= threshold. "
        "READ-ONLY."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "threshold_z": {"type": "number", "description": "Z-score threshold (default 2.0)", "default": 2.0},
            "lookback_days": {"type": "integer", "description": "Days for baseline average (default 7)", "default": 7},
            "limit": {"type": "integer", "description": "Max alerts to return", "default": 10},
        },
        "required": [],
    },
}

registry.register(
    name="pm_search_markets", toolset="prediction_markets", schema=PM_SEARCH_MARKETS_SCHEMA,
    handler=lambda args, **kw: pm_search_markets(
        query=args.get("query", ""), platform=args.get("platform", "both"), limit=args.get("limit", 10)),
    check_fn=_check_pm_requirements, emoji="🔍")

registry.register(
    name="pm_get_market", toolset="prediction_markets", schema=PM_GET_MARKET_SCHEMA,
    handler=lambda args, **kw: pm_get_market(
        ticker=args.get("ticker", ""), condition_id=args.get("condition_id", ""), platform=args.get("platform", "kalshi")),
    check_fn=_check_pm_requirements, emoji="📊")

registry.register(
    name="pm_get_orderbook", toolset="prediction_markets", schema=PM_GET_ORDERBOOK_SCHEMA,
    handler=lambda args, **kw: pm_get_orderbook(
        ticker=args.get("ticker", ""), token_id=args.get("token_id", ""), platform=args.get("platform", "kalshi"), depth=args.get("depth", 20)),
    check_fn=_check_pm_requirements, emoji="📈")

registry.register(
    name="pm_get_candlesticks", toolset="prediction_markets", schema=PM_GET_CANDLESTICKS_SCHEMA,
    handler=lambda args, **kw: pm_get_candlesticks(
        ticker=args.get("ticker", ""), token_id=args.get("token_id", ""), platform=args.get("platform", "kalshi"),
        interval=args.get("interval", "1day"), count=args.get("count", 30)),
    check_fn=_check_pm_requirements, emoji="🕯️")

registry.register(
    name="pm_get_price_history", toolset="prediction_markets", schema=PM_GET_PRICE_HISTORY_SCHEMA,
    handler=lambda args, **kw: pm_get_price_history(
        ticker=args.get("ticker", ""), token_id=args.get("token_id", ""), platform=args.get("platform", "kalshi"), days=args.get("days", 30)),
    check_fn=_check_pm_requirements, emoji="📉")

registry.register(
    name="pm_get_trades", toolset="prediction_markets", schema=PM_GET_TRADES_SCHEMA,
    handler=lambda args, **kw: pm_get_trades(
        ticker=args.get("ticker", ""), token_id=args.get("token_id", ""), platform=args.get("platform", "kalshi"), limit=args.get("limit", 20)),
    check_fn=_check_pm_requirements, emoji="💱")

registry.register(
    name="pm_cross_market_signals", toolset="prediction_markets", schema=PM_CROSS_MARKET_SIGNALS_SCHEMA,
    handler=lambda args, **kw: pm_cross_market_signals(
        query=args.get("query", ""), min_gap_pct=args.get("min_gap_pct", 1.0), limit=args.get("limit", 10)),
    check_fn=_check_pm_requirements, emoji="🔀")

registry.register(
    name="pm_volume_spike_scan", toolset="prediction_markets", schema=PM_VOLUME_SPIKE_SCAN_SCHEMA,
    handler=lambda args, **kw: pm_volume_spike_scan(
        threshold_z=args.get("threshold_z", 2.0), lookback_days=args.get("lookback_days", 7), limit=args.get("limit", 10)),
    check_fn=_check_pm_requirements, emoji="⚡")
