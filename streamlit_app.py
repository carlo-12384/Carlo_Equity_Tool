    # -*- coding: utf-8 -*-
"""Carlo Equity Tool — Streamlit App (Blocks-style UI)"""

import os, time, math, logging, textwrap, datetime as dt
import requests, pandas as pd, numpy as np, yfinance as yf
import matplotlib.pyplot as plt
from functools import lru_cache
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st
import json
import plotly.graph_objects as go # --- NEW --- Import Plotly


# -------------------- CONFIG / LOGGING --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

FINNHUB_KEY = os.getenv("FINNHUB_KEY", "d49tfm9r01qlaebj5lkgd49tfm9r01qlaebj5ll0")  # peers/profile/news
BASE = "https://finnhub.io/api/v1"

# -------------------- GENERIC HELPERS --------------------
def _safe(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(pd.to_numeric(x, errors="coerce"))
        except Exception:
            return np.nan

def _coerce_cols_desc(s: pd.Series) -> pd.Series:
    if not isinstance(s, pd.Series) or s.empty:
        return pd.Series([], dtype=float)
    idx = pd.to_datetime(s.index, errors="coerce")
    out = pd.Series(s.values, index=idx).dropna()
    return out.sort_index(ascending=False).astype(float)

def _first_row(df: pd.DataFrame, aliases) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series([], dtype=float)
    for k in aliases:
        if k in df.index:
            return df.loc[k]
    return pd.Series([], dtype=float)

def yf_quarterly(symbol: str):
    t = yf.Ticker(symbol)
    return t.quarterly_income_stmt, t.quarterly_balance_sheet, t.quarterly_cashflow

def ttm_from_rows(df: pd.DataFrame, aliases) -> float:
    try:
        s = _first_row(df, aliases)
        s = _coerce_cols_desc(s)
        return float(s.iloc[:4].sum()) if len(s) >= 4 else np.nan
    except Exception:
        return np.nan

def last_q_from_rows(df: pd.DataFrame, aliases) -> float:
    try:
        s = _first_row(df, aliases)
        s = _coerce_cols_desc(s)
        return float(s.iloc[0]) if len(s) >= 1 else np.nan
    except Exception:
        return np.nan

def load_revenue_series(symbol: str) -> pd.Series:
    aliases = [
        "Total Revenue","TotalRevenue","Revenue","Sales",
        "Operating Revenue","OperatingRevenue",
        "Net Revenue","Total Operating Revenue","Total Operating Revenues"
    ]
    t = yf.Ticker(symbol)
    for q_tbl_name in ["quarterly_income_stmt", "quarterly_financials", "quarterly_income_statement"]:
        q_df = getattr(t, q_tbl_name, None)
        s = _first_row(q_df, aliases)
        s = _coerce_cols_desc(s)
        if not s.empty:
            return s.astype(float)
    a_df = getattr(t, "income_stmt", None)
    s = _first_row(a_df, aliases)
    s = _coerce_cols_desc(s)
    return s.astype(float)
    
def generate_market_summary(index_data, macro_data=None, news_items=None):
    """
    Build a richer, human-style market wrap based on the index moves.
    Expects output from get_live_index_data().
    """
    if not index_data:
        return (
            "Market data is temporarily unavailable. "
            "Check your connection or try again in a few minutes."
        )

    # Pull core indices if present
    sp = next((x for x in index_data if x["symbol"] == "S&P 500"), None)
    nd = next((x for x in index_data if x["symbol"] == "NASDAQ"), None)
    dj = next((x for x in index_data if x["symbol"] == "Dow Jones"), None)

    lines = []

    # Overall tone
    pct_changes = [x["pct_change"] for x in index_data if isinstance(x.get("pct_change"), (int, float))]
    avg_move = np.nanmean(pct_changes) if pct_changes else 0.0
    if avg_move >= 0.7:
        tone = "risk-on, with buyers clearly in control"
    elif avg_move >= 0.2:
        tone = "constructive, with a mild positive bias"
    elif avg_move > -0.2:
        tone = "mixed and mostly sideways, with no dominant trend"
    elif avg_move > -0.7:
        tone = "cautious, as sellers have a slight edge"
    else:
        tone = "decidedly risk-off, with broad-based pressure across assets"

    lines.append(f"**Overall tone:** The market is {tone}.")

    # Index-level detail
    idx_bits = []
    for idx in [sp, nd, dj]:
        if idx:
            direction = "up" if idx["pct_change"] >= 0 else "down"
            idx_bits.append(f"{idx['symbol']} is {direction} {idx['pct_change']:.2f}%")

    if idx_bits:
        lines.append("**Major indices:** " + "; ".join(idx_bits) + ".")

    # Leadership / laggards
    leader = max(index_data, key=lambda x: x["pct_change"])
    laggard = min(index_data, key=lambda x: x["pct_change"])
    lines.append(
        f"**Leadership:** {leader['symbol']} is leading today, while "
        f"{laggard['symbol']} is lagging the field."
    )

    # Simple “style” read: growth vs value via NASDAQ vs Dow
    if nd and dj:
        style_spread = nd["pct_change"] - dj["pct_change"]
        if style_spread > 0.4:
            style_msg = "Growth/tech is outperforming more value-oriented names."
        elif style_spread < -0.4:
            style_msg = "Value / cyclicals are holding up better than growth and tech."
        else:
            style_msg = "Performance between growth and value looks fairly balanced."
        lines.append(f"**Style lens:** {style_msg}")

    # Volatility / conviction (based on move magnitude)
    max_abs_move = max(abs(x["pct_change"]) for x in index_data)
    if max_abs_move < 0.4:
        vol_msg = "Moves are relatively contained, suggesting low intraday volatility."
    elif max_abs_move < 1.0:
        vol_msg = "Indices are moving, but still within a normal volatility range."
    else:
        vol_msg = "Swings are elevated, pointing to higher-than-normal volatility."
    lines.append(f"**Volatility check:** {vol_msg}")

    if pct_changes:
        lines.append(f"**Average indexed move:** {avg_move:+.2f}% across the tracked indices.")

    strongest = weakest = None
    if macro_data:
        try:
            macro_sorted = sorted(macro_data, key=lambda x: x.get("change_val", 0.0))
            weakest = macro_sorted[0]
            strongest = macro_sorted[-1]
            macro_line = (
                f"**Macro cues:** {strongest['symbol']} {strongest['change_str']} vs "
                f"{weakest['symbol']} {weakest['change_str']}."
            )
            lines.append(macro_line)
        except Exception:
            strongest = weakest = None

    news_highlight = ""
    if news_items:
        primary = news_items[0]
        headline = primary.get("headline") or "Recent headlines"
        source = primary.get("publisher") or primary.get("source") or ""
        summary = primary.get("summary") or ""
        news_context = f"{headline}"
        if source:
            news_context += f" ({source})"
        news_highlight = news_context
        lines.append(f"**News trigger:** {news_context}")

    reason_parts = []
    if idx_bits:
        reason_parts.append(
            "Broad index behavior—led by "
            f"{leader['symbol']} and {laggard['symbol']}—suggests investors are rotating toward "
            "the pockets of strength while trimming the laggards."
        )
    if strongest and weakest:
        reason_parts.append(
            f"Macro cues are a reminder that {strongest['symbol']} is {strongest['change_str']} while "
            f"{weakest['symbol']} is {weakest['change_str']}, amplifying the directional bias for the risk trade."
        )
    if news_highlight:
        reason_parts.append(
            f"Headlines such as {news_highlight} are likely feeding the narrative, offering a plausible driver for the flow."
        )

    if reason_parts:
        reason_paragraph = " ".join(reason_parts)
        lines.append(f"**Why this matters:** {reason_paragraph}")

    # Wrap-up guidance style line
    lines.append(
        "**Takeaway for research:** It’s a good day to focus on how individual names "
        "are trading versus their sector and the broader tape, rather than just the headlines."
    )

    return "\n\n".join(lines)

MAJOR_NEWS_KEYWORDS = [
    "fed",
    "federal reserve",
    "interest rate",
    "rate hike",
    "inflation",
    "cpi",
    "ppi",
    "jobs report",
    "payroll",
    "recession",
    "stimulus",
    "gdp",
    "geopolitical",
    "opec",
    "oil prices",
    "treasury",
    "yield",
    "government shutdown",
    "debt ceiling",
    "default",
    "banking crisis",
    " earnings",
    "nasdaq",
    "s&p",
    "dow",
]

MARKET_INDEX_TICKERS = {"SPY", "QQQ", "DIA", "IWM", "^GSPC", "^IXIC", "^DJI", "^RUT"}
MAJOR_NEWS_CATEGORIES = ["general", "forex", "crypto"]

def _is_market_moving_headline(headline: str, summary: str = "", related: str = "") -> bool:
    text = f"{headline} {summary}".lower()
    if any(kw in text for kw in MAJOR_NEWS_KEYWORDS):
        return True
    rel = (related or "").replace(" ", "").upper()
    if not rel:
        return False
    related_syms = {sym for sym in rel.split(",") if sym}
    return bool(related_syms.intersection(MARKET_INDEX_TICKERS))

def _parse_finnhub_news_item(item: dict):
    title = (item.get("headline") or "").strip()
    link = item.get("url") or ""
    if not title or not link:
        return None
    summary = (item.get("summary") or "").strip()
    related = item.get("related") or ""
    ts = pd.to_datetime(item.get("datetime"), unit="s", errors="coerce")
    is_major = _is_market_moving_headline(title, summary, related)
    return {
        "headline": title,
        "url": link,
        "publisher": item.get("source") or item.get("category") or "",
        "time": ts,
        "summary": summary,
        "is_major": is_major,
    }

@st.cache_data(ttl=300)
def get_market_news(n=8):
    """
    Fetch broad market news with an emphasis on macro / market-moving headlines.
    Primary source: Finnhub multi-category feed, fallback to SPY news from yfinance.
    """
    items = []
    seen = set()

    for category in MAJOR_NEWS_CATEGORIES:
        raw = safe_finnhub_get("/news", category=category) or []
        for item in raw:
            parsed = _parse_finnhub_news_item(item)
            if not parsed or not parsed["is_major"]:
                continue
            key = parsed["url"]
            if key in seen:
                continue
            seen.add(key)
            items.append(parsed)

    if len(items) < n:
        try:
            spy_news = yf.Ticker("SPY").news or []
        except Exception:
            spy_news = []
        for item in spy_news:
            title = item.get("title") or item.get("headline") or ""
            link = item.get("link") or item.get("url") or ""
            if not title or not link or link in seen:
                continue
            summary = (item.get("summary") or "").strip()
            ts_raw = item.get("providerPublishTime") or item.get("datetime")
            ts = pd.to_datetime(ts_raw, unit="s", errors="coerce")
            is_major = _is_market_moving_headline(title, summary, related="SPY")
            if not is_major:
                continue
            seen.add(link)
            items.append(
                {
                    "headline": title,
                    "url": link,
                    "publisher": item.get("publisher") or item.get("source") or "",
                    "time": ts,
                    "summary": summary,
                    "is_major": True,
                }
            )
            if len(items) >= n:
                break

    items.sort(key=lambda x: x.get("time") or pd.Timestamp.min, reverse=True)
    cleaned = []
    for item in items[:n]:
        cleaned.append(
            {
                "headline": item["headline"],
                "url": item["url"],
                "publisher": item.get("publisher") or "",
                "time": item.get("time"),
                "is_major": True,
                "summary": item.get("summary", ""),
            }
        )
    return cleaned


def get_economic_calendar():
    return [
        ("CPI Report", "Wednesday 8:30 AM"),
        ("FOMC Minutes", "Thursday 2:00 PM"),
        ("Initial Jobless Claims", "Thursday 8:30 AM"),
        ("PMI Index", "Friday 9:45 AM"),
    ]
    
def get_smart_money_signals():
    return {
        "Insider Buy/Sell Ratio": "1.8 (bullish)",
        "Institutional Accumulation": "Moderate inflows",
        "Hedge Fund Sentiment": "Net long positioning rising",
        "ETF Money Flow": "$2.4B inflow over 5 days",
    }
    
def get_earnings_momentum_data():
    sectors = ["Tech","Health","Energy","Financials","Consumer","Industrials"]
    beats = np.random.uniform(50, 85, size=6)  # placeholder synthetic
    misses = 100 - beats
    return sectors, beats, misses

def radar_chart(sectors, beats):
    labels = sectors
    stats = beats

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Earnings Momentum Radar")
    return fig


# -------------------- PRICE / RETURNS --------------------
def calculate_vwap(ticker: str):
    """5d VWAP using daily data (approx)."""
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if hist is None or hist.empty:
            return None, None, pd.DataFrame()
        hist = hist.dropna(subset=["Close", "Volume"])
        if hist.empty or hist["Volume"].sum() == 0:
            return None, None, hist
        vwap = (hist["Close"] * hist["Volume"]).sum() / hist["Volume"].sum()
        latest = float(hist["Close"].iloc[-1])
        return latest, float(vwap), hist
    except Exception:
        return None, None, pd.DataFrame()

def get_total_return(symbol: str, months=12):
    """Simple total return over N months via yfinance."""
    try:
        period = f"{months}mo" if months in (1,3,6,12,24) else "1y"
        h = yf.Ticker(symbol).history(period=period, interval="1d")
        if h.empty or h["Close"].dropna().shape[0] < 5:
            return None
        s, e = float(h["Close"].dropna().iloc[0]), float(h["Close"].dropna().iloc[-1])
        return (e / s) - 1.0 if s > 0 else None
    except Exception:
        return None

def get_price_and_shares(symbol: str):
    """Robust price & shares discovery."""
    t = yf.Ticker(symbol)
    fast = t.fast_info or {}
    price = _safe(fast.get("last_price"))
    if not np.isfinite(price):
        try:
            h = t.history(period="5d", interval="1d")
            if not h.empty:
                price = float(h["Close"].dropna().iloc[-1])
        except Exception:
            price = np.nan
    shares = _safe(fast.get("shares"))
    if not np.isfinite(shares):
        try:
            q_is, _, _ = yf_quarterly(symbol)
            shares = last_q_from_rows(q_is, [
                "Diluted Average Shares","Basic Average Shares",
                "Weighted Average Shs Out Dil","Weighted Average Shares Diluted",
                "Weighted Average Shs Out","Weighted Average Shares",
            ])
        except Exception:
            shares = np.nan
    if not np.isfinite(shares):
        try:
            info = t.get_info()
            shares = _safe(info.get("sharesOutstanding"))
        except Exception:
            pass
    return price, shares

# --- MODIFIED FUNCTION ---
# Changed to fetch the 4 indices from the screenshot
@st.cache_data(ttl=60) # Cache for 1 minute
def get_live_index_data():
    """
    Fetches live price data and daily change for major indices.
    """
    indices = {
        '^GSPC': 'S&P 500',
        '^IXIC': 'NASDAQ',
        '^DJI': 'Dow Jones',
        '^RUT': 'Russell 2000' # --- NEW ---
    }
    data = []
    for ticker, name in indices.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d", interval="1d")
            if hist.empty or len(hist) < 2:
                # Try fast_info as a fallback
                fast = yf.Ticker(ticker).fast_info
                last_price = fast.get("last_price")
                prev_close = fast.get("previous_close")
                if not last_price or not prev_close:
                    continue
            else:
                prev_close = hist['Close'].iloc[0]
                last_price = hist['Close'].iloc[-1]
            
            change = last_price - prev_close
            pct_change = (change / prev_close) * 100
            
            data.append({
                'symbol': name,
                'price': last_price,
                'change': change,
                'pct_change': pct_change
            })
        except Exception as e:
            logging.warning(f"Failed to get live index data for {ticker}: {e}")
    return data
    
# --- Retrieving Macro Data ---
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_global_macro_data():
    """
    Fetches live price data and daily change for key macro assets.
    ALWAYS returns all assets so the ticker tape stays full, even
    if some data is missing.
    """
    assets = {
        '^TNX': '10-Yr Yield',
        'CL=F': 'Crude Oil',
        'GC=F': 'Gold',
        'AGG':  'US Bonds (AGG)',
    }

    data = []

    for ticker, name in assets.items():
        last_price = None
        prev_close = None

        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d", interval="1d")

            if hist is not None and not hist.empty and len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[0])
                last_price = float(hist["Close"].iloc[-1])
            else:
                fast = t.fast_info or {}
                last_price = fast.get("last_price") or fast.get("lastPrice")
                prev_close = fast.get("previous_close") or fast.get("previousClose")

                if last_price is not None:
                    last_price = float(last_price)
                if prev_close is not None:
                    prev_close = float(prev_close)
        except Exception as e:
            logging.warning(f"Failed to get macro data for {ticker}: {e}")

        # ---- Fallbacks so we NEVER drop an item ----
        if last_price is None and prev_close is None:
            unit = "%" if ticker == "^TNX" else "$"
            price_str = f"N/A{unit}"
            change_str = "+0.00 (+0.00%)"
            change_val = 0.0
        else:
            if last_price is None:
                last_price = prev_close
            if prev_close is None or prev_close == 0:
                prev_close = last_price

            try:
                change = last_price - prev_close
                pct_change = (change / prev_close) * 100 if prev_close else 0.0
            except Exception:
                change = 0.0
                pct_change = 0.0

            unit = "%" if ticker == "^TNX" else "$"
            price_str = f"{last_price:,.2f}{unit}"
            change_str = f"{change:+.2f} ({pct_change:+.2f}%)"
            change_val = change

        data.append(
            {
                "symbol": name,
                "price_str": price_str,
                "change_str": change_str,
                "change_val": change_val,
            }
        )

    return data
    
@st.cache_data(ttl=60)  # Refresh every 60s
def get_dashboard_kpis():
    """
    Macro KPIs for the header bar.
    Uses assets that are NOT already on the page:
      - VIX (volatility)
      - US Dollar Index (UUP as proxy)
      - High Yield Credit (HYG)
      - Investment Grade Credit (LQD)
    """
    kpi_assets = {
        "^VIX": "Volatility (VIX)",
        "UUP":  "US Dollar (UUP)",
        "HYG":  "High Yield Credit",
        "LQD":  "IG Credit",
    }

    out = []
    for ticker, label in kpi_assets.items():
        last_price = None
        prev_close = None

        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d", interval="1d")
            if hist is not None and not hist.empty and len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[0])
                last_price = float(hist["Close"].iloc[-1])
            else:
                fast = t.fast_info or {}
                last_price = fast.get("last_price") or fast.get("lastPrice")
                prev_close = fast.get("previous_close") or fast.get("previousClose")
                if last_price is not None:
                    last_price = float(last_price)
                if prev_close is not None:
                    prev_close = float(prev_close)
        except Exception as e:
            logging.warning(f"Failed to get KPI data for {ticker}: {e}")

        if last_price is None and prev_close is None:
            value_str = "N/A"
            change_str = "+0.00 (0.00%)"
            change_val = 0.0
        else:
            if last_price is None:
                last_price = prev_close
            if prev_close is None or prev_close == 0:
                prev_close = last_price

            try:
                change = last_price - prev_close
                pct_change = (change / prev_close) * 100 if prev_close else 0.0
            except Exception:
                change = 0.0
                pct_change = 0.0

            value_str = f"{last_price:,.2f}"
            change_str = f"{change:+.2f} ({pct_change:+.2f}%)"
            change_val = change

        out.append(
            {
                "label": label,
                "value_str": value_str,
                "change_str": change_str,
                "change_val": change_val,
            }
        )

    return out
    
@st.cache_data(ttl=300)
def get_macro_indicator_cards():
    """
    Returns 4 macro cards:
      - Volatility (VIX)
      - US Dollar (UUP)
      - High Yield Credit (HYG)
      - IG Credit (LQD)
    Each card has: label, last value, change, pct change.
    """
    tickers = {
        "^VIX": "Volatility (VIX)",
        "UUP": "US Dollar (UUP)",
        "HYG": "High Yield Credit",
        "LQD": "IG Credit",
    }

    cards = []
    for tkr, label in tickers.items():
        try:
            hist = yf.Ticker(tkr).history(period="2d", interval="1d")
            if hist is None or hist.empty or len(hist) < 2:
                continue

            prev = float(hist["Close"].iloc[0])
            last = float(hist["Close"].iloc[-1])
            change = last - prev
            pct = (change / prev) * 100 if prev != 0 else 0.0

            cards.append(
                {
                    "label": label,
                    "value": last,
                    "change": change,
                    "pct": pct,
                }
            )
        except Exception as e:
            logging.warning(f"Failed macro card for {tkr}: {e}")
            continue

    return cards


@st.cache_data(ttl=300)
def get_index_card_metrics(ticker: str):
    """
    Robust index metrics based purely on historical prices.
    Works for indices like ^DJI, ^IXIC, ^GSPC, ^RUT.
    
    Returns:
      {
        "last": float,
        "change": float,
        "change_pct": float,
        "metrics": [
            {"label": "YTD Performance", "value": "..."},
            {"label": "Avg. Volume", "value": "..."},
            {"label": "52-Wk Range", "value": "..."},
        ],
      }
    or None if data could not be fetched at all.
    """
    import datetime as dt

    try:
        today = dt.date.today()
        year_start = dt.date(today.year, 1, 1)

        # 1) YTD history (for last, prev close, YTD %)
        ytd = yf.download(
            ticker,
            start=year_start,
            interval="1d",
            progress=False,
        )

        # 2) 1-year history (for 52-week range + avg volume)
        one_year = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
        )
    except Exception as e:
        logging.warning(f"Failed to download index data for {ticker}: {e}")
        return None

    if ytd is None or ytd.empty or one_year is None or one_year.empty:
        logging.warning(f"No price history returned for {ticker}.")
        return None

    try:
        # ----- Current price & daily change -----
        last_close = float(ytd["Close"].iloc[-1])

        if len(ytd) >= 2:
            prev_close = float(ytd["Close"].iloc[-2])
            day_change = last_close - prev_close
            day_change_pct = (day_change / prev_close) * 100.0
        else:
            day_change = 0.0
            day_change_pct = 0.0

        # ----- YTD performance -----
        first_ytd_close = float(ytd["Close"].iloc[0])
        ytd_pct = (last_close / first_ytd_close - 1.0) * 100.0

        # ----- 52-week range -----
        low_52w = float(one_year["Low"].min())
        high_52w = float(one_year["High"].max())

        # ----- Average volume (last 1y) -----
        avg_volume = int(one_year["Volume"].mean())

        metrics = [
            {
                "label": "YTD Performance",
                "value": f"{ytd_pct:+.2f}%",  # + sign for positive
            },
            {
                "label": "Avg. Volume",
                "value": f"{avg_volume:,.0f}",
            },
            {
                "label": "52-Wk Range",
                "value": f"{low_52w:,.2f} - {high_52w:,.2f}",
            },
        ]

        return {
            "last": last_close,
            "change": day_change,
            "change_pct": day_change_pct,
            "metrics": metrics,
        }

    except Exception as e:
        logging.warning(f"Failed to compute metrics for {ticker}: {e}")
        return None


# --- NEW FUNCTION ---
@st.cache_data(ttl=300)
def get_sector_performance():
    """
    Fetches daily performance for the 11 main S&P 500 sector ETFs.
    """
    sectors = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLC': 'Communication',
        'XLY': 'Consumer Discr.',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLI': 'Industrials'
    }
    
    try:
        data = yf.Tickers(list(sectors.keys())).history(period="2d", interval="1d")
        if data.empty:
            return {}
        
        perf = {}
        for ticker, name in sectors.items():
            try:
                close_prices = data[('Close', ticker)].dropna()
                if len(close_prices) >= 2:
                    pct_change = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
                    perf[name] = pct_change * 100
                else:
                    perf[name] = np.nan # --- MODIFICATION --- Not enough data
            except Exception:
                perf[name] = np.nan # --- MODIFICATION --- Error on this ticker
        return perf
    except Exception as e:
        logging.warning(f"Failed to get sector performance data: {e}")
        return {}


# --- NEW FUNCTION ---
def plot_sector_heatmap(sector_data: dict):
    """
    Plotly Treemap of sector performance with a red→grey→green gradient.
    Tiles are equal-sized; color encodes 1-day % change.
    """
    if not sector_data:
        fig = go.Figure()
        fig.update_layout(
            title_text="Sector Performance (Data Unavailable)",
            title_x=0.5,
            margin=dict(t=40, l=4, r=4, b=4),
        )
        return fig

    labels = list(sector_data.keys())
    perfs = []

    for v in sector_data.values():
        try:
            perf = float(v)
            if not np.isfinite(perf):
                perf = 0.0
        except (ValueError, TypeError):
            perf = 0.0
        perfs.append(perf)

    # Equal tile areas; color carries the meaning
    values = [1.0] * len(labels)
    customdata = [f"{p:+.2f}%" for p in perfs]

    # Make sure we always have a sensible center around 0
    max_abs = max(max(abs(p) for p in perfs), 1e-6)

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=[""] * len(labels),
            values=values,
            customdata=customdata,
            texttemplate="<b>%{label}</b><br>%{customdata}",
            hovertemplate="<b>%{label}</b><br>1D Change: %{customdata}<extra></extra>",
            marker=dict(
                colors=perfs,
                colorscale=[
                    [0.0, "#7F1D1D"],   # deep red
                    [0.5, "#1F2933"],   # neutral dark grey
                    [1.0, "#047857"],   # deep green
                ],
                cmid=0,  # center colorscale at 0%
            ),
        )
    )

    fig.update_layout(
        margin=dict(t=40, l=4, r=4, b=4),
        title=dict(text="Daily Sector Performance Heatmap", x=0.5, font=dict(size=18)),
    )
    return fig


# -------------------- FINNHUB (FREE) --------------------
def safe_finnhub_get(path, **params):
    try:
        params["token"] = FINNHUB_KEY
        r = requests.get(f"{BASE}{path}", params=params, timeout=12)
        if r.status_code == 429:
            logging.warning("Finnhub 429 rate-limit")
        r.raise_for_status()
        data = r.json()
        if data is None or (isinstance(data, dict) and not data) or (isinstance(data, list) and len(data) == 0):
            return None
        return data
    except Exception as e:
        logging.warning(f"Finnhub error {path} {params.get('symbol')}: {e}")
        return None

@lru_cache(maxsize=512)
def get_profile(symbol: str) -> dict:
    try:
        return safe_finnhub_get("/stock/profile2", symbol=symbol) or {}
    except Exception:
        return {}

def get_peers(ticker: str, max_peers=10):
    data = safe_finnhub_get("/stock/peers", symbol=ticker)
    if not data:
        return []
    peers = [p for p in data if isinstance(p, str) and p.upper() != ticker.upper()]
    return peers[:max_peers]

def filter_peers_industry_size(ticker: str, peers: list, size_band=(0.25, 4.0)):
    prof = get_profile(ticker) or {}
    ind = (prof.get("finnhubIndustry") or "").lower()
    mc  = prof.get("marketCapitalization") or None
    if not peers:
        return []
    kept = []
    for p in peers:
        pp = get_profile(p) or {}
        ok_ind = (pp.get("finnhubIndustry") or "").lower() == ind if ind else True
        pmc = pp.get("marketCapitalization")
        ok_mc = True if (mc is None or pmc is None) else (size_band[0]*mc <= pmc <= size_band[1]*mc)
        if ok_ind and ok_mc:
            kept.append(p)
    return kept or peers

def corr_peers(ticker: str, candidates: list, lookback_days=365, top_k=6):
    """Select top-K by return correlation over last ~1y daily data."""
    try:
        base = yf.Ticker(ticker).history(period="1y", interval="1d")["Close"].pct_change().dropna()
        if base.empty:
            return []
    except Exception:
        return []
    cors = []
    for c in candidates:
        try:
            h = yf.Ticker(c).history(period="1y", interval="1d")["Close"].pct_change().dropna()
            j = base.index.intersection(h.index)
            if len(j) >= 60:
                cors.append((c, float(np.corrcoef(base.loc[j], h.loc[j])[0, 1])))
        except Exception:
            continue
    cors = sorted(cors, key=lambda x: -x[1])
    return [c for c, _ in cors[:top_k]]

def build_peerset(ticker: str, vendor_max=20, final_cap=8):
    raw = get_peers(ticker, max_peers=vendor_max) or []
    raw = [p.upper() for p in raw if isinstance(p, str) and p.upper() != ticker.upper()]
    filt = filter_peers_industry_size(ticker, raw)
    corr = corr_peers(ticker, filt or raw, top_k=final_cap*2)

    peers = []
    inter = [p for p in filt if p in corr]
    peers.extend(inter)

    for p in corr:
        if p not in peers:
            peers.append(p)
    for p in filt:
        if p not in peers:
            peers.append(p)
    for p in raw:
        if p not in peers:
            peers.append(p)

    peers = [p for p in peers if p != ticker.upper()][:final_cap]
    return peers

# -------------------- COMPANY SUMMARY & NEWS --------------------
def get_company_text_summary(symbol: str, row: dict) -> str:
    prof = get_profile(symbol) or {}
    name = prof.get("name") or symbol.upper()
    ind = prof.get("finnhubIndustry") or "—"
    mc  = row.get("MarketCap") if isinstance(row, (dict, pd.Series, dict)) else None
    exch= prof.get("exchange") or prof.get("ticker") or ""
    mc_txt = "N/A"
    if isinstance(mc, (int, float)) and np.isfinite(mc) and mc > 0:
        if mc >= 1000:
            mc_txt = f"${mc/1000:,.1f}B"
        else:
            mc_txt = f"${mc:,.0f}M"
    return (
        f"**{name}** ({symbol.upper()}) — {ind} on {exch}. "
        f"Est. market cap {mc_txt}."
    )

def get_company_metrics_summary(row: dict) -> str:
    bits = []
    gm = row.get("GrossMargin%") if isinstance(row, (dict, pd.Series)) else np.nan
    em = row.get("EBITDAMargin%") if isinstance(row, (dict, pd.Series)) else np.nan
    rg = row.get("RevenueGrowth%") if isinstance(row, (dict, pd.Series)) else np.nan
    roe = row.get("ROE%") if isinstance(row, (dict, pd.Series)) else np.nan
    ev_ebitda = row.get("EV/EBITDA") if isinstance(row, (dict, pd.Series)) else np.nan
    dte = row.get("DebtToEquity") if isinstance(row, (dict, pd.Series)) else np.nan
    pe_raw = row.get("P/E (raw)") if isinstance(row, (dict, pd.Series)) else np.nan
    pb_raw = row.get("P/B (raw)") if isinstance(row, (dict, pd.Series)) else np.nan
    ps_raw = row.get("P/S (raw)") if isinstance(row, (dict, pd.Series)) else np.nan

    def format_metric(value, prefix="", suffix="", is_percent=False, invert_color=False):
        if value is None or not np.isfinite(value):
            return ""
        color_class = "positive-metric" if value >= 0 else "negative-metric"
        if invert_color:
            color_class = "negative-metric" if value >= 0 else "positive-metric"
        format_str = f"{{:.1f}}{{}}" if not is_percent else f"{{:+.1f}}{{}}"
        return f"<span class=\"{color_class}\">" + format_str.format(value, suffix) + "</span>"

    if pe_raw is not None and np.isfinite(pe_raw): bits.append(f"- P/E: {format_metric(pe_raw, suffix='x', invert_color=True)}")
    if pb_raw is not None and np.isfinite(pb_raw): bits.append(f"- P/B: {format_metric(pb_raw, suffix='x', invert_color=True)}")
    if ps_raw is not None and np.isfinite(ps_raw): bits.append(f"- P/S: {format_metric(ps_raw, suffix='x', invert_color=True)}")
    if ev_ebitda is not None and np.isfinite(ev_ebitda): bits.append(f"- EV/EBITDA: {format_metric(ev_ebitda, suffix='x', invert_color=True)}")
    if gm is not None and np.isfinite(gm): bits.append(f"- Gross Margin: {format_metric(gm, suffix='%', is_percent=True)}")
    if em is not None and np.isfinite(em): bits.append(f"- EBITDA Margin: {format_metric(em, suffix='%', is_percent=True)}")
    if rg is not None and np.isfinite(rg): bits.append(f"- Revenue Growth (YoY, TTM): {format_metric(rg, suffix='%', is_percent=True)}")
    if roe is not None and np.isfinite(roe): bits.append(f"- ROE: {format_metric(roe, suffix='%', is_percent=True)}")
    if dte is not None and np.isfinite(dte): bits.append(f"- Debt/Equity: {format_metric(dte, suffix='x', invert_color=True)}")

    return "\n".join(bits) if bits else "_Limited numerical metrics available._"

def get_company_news(symbol: str, n=8):
    """Return list of dicts: {headline, url, source, datetime, summary?} using Finnhub, fallback yfinance."""
    out = []
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=14)
        data = safe_finnhub_get("/company-news", symbol=symbol.upper(), _from=start.isoformat(), to=end.isoformat())
        if data:
            for item in sorted(data, key=lambda x: x.get("datetime", 0), reverse=True)[:n]:
                out.append({
                    "headline": item.get("headline") or "",
                    "url": item.get("url") or "",
                    "source": item.get("source") or "",
                    "datetime": pd.to_datetime(item.get("datetime"), unit="s", errors="coerce"),
                    "summary": item.get("summary") or "",
                })
    except Exception:
        pass
    if len(out) < 3:
        try:
            news = yf.Ticker(symbol).news or []
            for item in news[: (n - len(out))]:
                out.append({
                    "headline": item.get("title") or "",
                    "url": item.get("link") or "",
                    "source": item.get("publisher") or "",
                    "datetime": pd.to_datetime(item.get("providerPublishTime"), unit="s", errors="coerce"),
                    "summary": "",
                })
        except Exception:
            pass
    return out[:n]

def render_news_md(news_list):
    if not news_list:
        return "_No recent headlines available (last 14 days)._"
    lines = []
    for it in news_list:
        ts = ""
        if isinstance(it.get("datetime"), pd.Timestamp) and pd.notna(it["datetime"]):
            ts = it["datetime"].strftime("%Y-%m-%d %H:%M")
        src = f" — {it.get('source')}" if it.get('source') else ""
        h = it.get("headline"," ").replace("\n"," ").strip()
        url = it.get("url"," ")
        sumline = it.get("summary"," ").strip()
        if sumline:
            sumline = "  \n<small>" + textwrap.shorten(sumline, width=220, placeholder="…") + "</small>"
        lines.append(f"- [{h}]({url}) <small>{src} • {ts}</small>{sumline}")
    return "\n".join(lines)

# -------------------- FUNDAMENTALS (FREE) --------------------
def get_metrics(symbol: str) -> dict:
    """
    Free-only fundamentals computed from yfinance statements (TTM where possible).
    Expanded factor set + industry tagging for neutralization.
    """
    out = {"Ticker": symbol}
    try:
        t = yf.Ticker(symbol)
        price, shares = get_price_and_shares(symbol)

        fast = t.fast_info or {}
        mktcap = _safe(fast.get("market_cap"))
        if (not np.isfinite(mktcap)) and np.isfinite(price) and np.isfinite(shares) and shares > 0:
            mktcap = price * shares
        if not np.isfinite(mktcap):
            try:
                info = t.get_info()
                mktcap = _safe(info.get("marketCap"))
            except Exception:
                pass

        q_is, q_bs, q_cf = yf_quarterly(symbol)

        rev_ttm    = ttm_from_rows(q_is, ["Total Revenue","TotalRevenue","Revenue"])
        cogs_ttm   = ttm_from_rows(q_is, ["Cost Of Revenue","CostOfRevenue"])
        gross_ttm  = (rev_ttm - cogs_ttm) if all(np.isfinite([rev_ttm, cogs_ttm])) else np.nan
        ebit_ttm   = ttm_from_rows(q_is, ["Ebit","EBIT","Operating Income","OperatingIncome"])
        dep_ttm    = ttm_from_rows(q_cf, ["Depreciation","Depreciation And Amortization","Depreciation Amortization Depletion"])
        ebitda_ttm = (ebit_ttm + dep_ttm) if all(np.isfinite([ebit_ttm, dep_ttm])) else np.nan
        netinc_ttm = ttm_from_rows(q_is, [
            "Net Income Common Stockholders",
            "Net Income Applicable To Common Shares",
            "Net Income","NetIncome"
        ])
        cfo_ttm = ttm_from_rows(q_cf, ["Operating Cash Flow","Total Cash From Operating Activities"])
        capex_ttm = ttm_from_rows(q_cf, ["Capital Expenditure","Change In Property Plant Equipment"])
        fcff_ttm = np.nan
        if np.isfinite(cfo_ttm):
            if np.isfinite(capex_ttm):
                fcff_ttm = cfo_ttm - abs(capex_ttm)
            else:
                fcff_ttm = cfo_ttm
        fcf_margin_pct = np.nan
        if np.isfinite(fcff_ttm) and np.isfinite(rev_ttm) and rev_ttm != 0:
            fcf_margin_pct = (fcff_ttm / rev_ttm) * 100

        rev_series_q = load_revenue_series(symbol)
        if not rev_series_q.empty and len(rev_series_q) >= 8:
            rev_prev_ttm = float(rev_series_q.iloc[4:8].sum())
        else:
            rev_prev_ttm = np.nan

        net_series_q = _coerce_cols_desc(_first_row(q_is, [
            "Net Income Common Stockholders","Net Income Applicable To Common Shares","Net Income","NetIncome"
        ]))
        if not net_series_q.empty and len(net_series_q) >= 8:
            net_prev_ttm = float(net_series_q.iloc[4:8].sum())
        else:
            net_prev_ttm = np.nan
        _ = net_prev_ttm  # currently unused

        rev_growth_pct = np.nan
        try:
            if np.isfinite(rev_ttm) and np.isfinite(rev_prev_ttm) and rev_prev_ttm != 0:
                rev_growth_pct = ((rev_ttm / rev_prev_ttm) - 1.0) * 100
            if np.isnan(rev_growth_pct):
                a_is = getattr(t, "income_stmt", None)
                annual_rev_series = _coerce_cols_desc(_first_row(a_is, [
                    "Total Revenue","TotalRevenue","Revenue","Sales",
                    "Operating Revenue","OperatingRevenue","Net Revenue",
                    "Total Operating Revenue","Total Operating Revenues"
                ]))
                if not annual_rev_series.empty and len(annual_rev_series) >= 2:
                    curr_annual_rev, prev_annual_rev = float(annual_rev_series.iloc[0]), float(annual_rev_series.iloc[1])
                    if np.isfinite(curr_annual_rev) and np.isfinite(prev_annual_rev) and prev_annual_rev != 0:
                        rev_growth_pct = ((curr_annual_rev / prev_annual_rev) - 1.0) * 100
        except Exception:
            rev_growth_pct = np.nan

        total_debt = last_q_from_rows(q_bs, ["Total Debt","TotalDebt"])
        if not np.isfinite(total_debt):
            lt = last_q_from_rows(q_bs, ["Long Term Debt","LongTermDebt","Long Term Debt Noncurrent"])
            st_ = last_q_from_rows(q_bs, ["Short Long Term Debt","Short Long Term Debt Total","Short/Current Long Term Debt"])
            total_debt = (lt if np.isfinite(lt) else 0) + (st_ if np.isfinite(st_) else 0)
            if total_debt == 0:
                total_debt = np.nan

        cash = last_q_from_rows(q_bs, [
            "Cash And Cash Equivalents","Cash",
            "Cash And Cash Equivalents And Short Term Investments"
        ])

        total_equity = last_q_from_rows(q_bs, ["Stockholders Equity","Total Stockholder Equity","Total Equity"])

        assets_key = None
        for k in ["Total Assets","TotalAssets","Assets"]:
            if isinstance(q_bs, pd.DataFrame) and k in q_bs.index:
                assets_key = k
                break
        assets_series = _coerce_cols_desc(q_bs.loc[assets_key]) if assets_key else pd.Series([], dtype=float)
        assets_avg = float(assets_series.iloc[:4].mean()) if len(assets_series) >= 2 else np.nan
        asset_growth_pct = np.nan
        if len(assets_series) >= 2 and assets_series.iloc[1] != 0:
            asset_growth_pct = ((assets_series.iloc[0] / assets_series.iloc[1]) - 1.0) * 100

        interest_exp_ttm = ttm_from_rows(q_is, ["Interest Expense","InterestExpense"])
        if np.isfinite(interest_exp_ttm) and interest_exp_ttm != 0:
            interest_coverage = ebit_ttm / abs(interest_exp_ttm) if np.isfinite(ebit_ttm) else np.nan
        else:
            interest_coverage = np.nan

        gross_profitability = (gross_ttm / assets_avg) if all(np.isfinite([gross_ttm, assets_avg])) and assets_avg != 0 else np.nan

        def delta_last(series):
            return float(series.iloc[0] - series.iloc[1]) if len(series) >= 2 else np.nan
        CA = _coerce_cols_desc(_first_row(q_bs, ["Current Assets","Total Current Assets","CurrentAssets"]))
        CL = _coerce_cols_desc(_first_row(q_bs, ["Current Liabilities","Total Current Liabilities","CurrentLiabilities"]))
        CASH = _coerce_cols_desc(_first_row(q_bs, ["Cash And Cash Equivalents","Cash"]))
        STD = _coerce_cols_desc(_first_row(q_bs, ["Short Long Term Debt","Short/Current Long Term Debt","Short Long Term Debt Total"]))
        dCA = delta_last(CA); dCL = delta_last(CL); dCASH = delta_last(CASH); dSTD = delta_last(STD)
        accruals_pct = np.nan
        try:
            if all(np.isfinite([dCA, dCASH, dCL, dSTD, dep_ttm, assets_avg])) and assets_avg != 0:
                acc = ((dCA - dCASH) - (dCL - dSTD) - dep_ttm)
                accruals_pct = (acc / assets_avg) * 100.0
        except Exception:
            pass

        ev = (mktcap + total_debt - cash) if all(np.isfinite([mktcap, total_debt, cash])) else np.nan
        eps_ttm = (netinc_ttm / shares) if all(np.isfinite([netinc_ttm, shares])) and shares > 0 else np.nan

        pe_raw = np.nan
        if np.isfinite(eps_ttm) and eps_ttm > 0 and np.isfinite(price):
            pe_raw = price / eps_ttm

        pb_raw  = (mktcap / total_equity) if all(np.isfinite([mktcap, total_equity])) and total_equity > 0 else np.nan
        ps_raw  = (mktcap / rev_ttm) if all(np.isfinite([mktcap, rev_ttm])) and rev_ttm > 0 else np.nan
        ev_ebitda_raw = (ev / ebitda_ttm) if all(np.isfinite([ev, ebitda_ttm])) and ebitda_ttm > 0 else np.nan

        gross_margin_pct  = (gross_ttm  / rev_ttm) * 100 if all(np.isfinite([gross_ttm, rev_ttm])) and rev_ttm != 0 else np.nan
        ebitda_margin_pct = (ebitda_ttm / rev_ttm) * 100 if all(np.isfinite([ebitda_ttm, rev_ttm])) and rev_ttm != 0 else np.nan
        roe_pct = (netinc_ttm / total_equity) * 100 if all(np.isfinite([netinc_ttm, total_equity])) and total_equity != 0 else np.nan
        dte = (total_debt / total_equity) * 100 if all(np.isfinite([total_debt, total_equity])) and total_equity != 0 else np.nan
        debt_ebitda = (total_debt / ebitda_ttm) if all(np.isfinite([total_debt, ebitda_ttm])) and ebitda_ttm != 0 else np.nan

        prof = get_profile(symbol) or {}
        industry = prof.get("finnhubIndustry") or ""

        out.update({
            "MarketCap": mktcap, "Latest Price": price, "EPS_TTM": eps_ttm,
            "P/E (raw)": pe_raw, "P/B (raw)": pb_raw, "EV/EBITDA (raw)": ev_ebitda_raw, "P/S (raw)": ps_raw,
            "P/E Ratio": pe_raw, "P/B Ratio": pb_raw, "EV/EBITDA": ev_ebitda_raw, "Price/Sales": ps_raw,
            "ROE%": roe_pct, "GrossMargin%": gross_margin_pct, "EBITDAMargin%": ebitda_margin_pct,
            "FCF Margin%": fcf_margin_pct, "FCF_TTM": fcff_ttm,
            "RevenueGrowth%": rev_growth_pct, "DebtToEquity": dte,
            "Debt/EBITDA": debt_ebitda, "Total Debt": total_debt, "EBITDA_TTM": ebitda_ttm,
            "GrossProfitability": gross_profitability, "AssetGrowth%": asset_growth_pct,
            "Accruals%": accruals_pct, "InterestCoverage": interest_coverage,
            "Industry": industry
        })
    except Exception as e:
        logging.warning(f"get_metrics failed for {symbol}: {e}")
    return out

# -------------------- FACTOR PIPELINE --------------------
FACTOR_BUCKETS = {
    "Valuation":  ["P/E Ratio","P/B Ratio","EV/EBITDA","Price/Sales"],
    "Quality":    ["ROE%","GrossMargin%","EBITDAMargin%","InterestCoverage"],
    "Growth":     ["RevenueGrowth%","AssetGrowth%"],
    "Momentum":   ["TTM-Return","Mom_VWAP_Diff%"],
    "Leverage":   ["DebtToEquity"],
    "Efficiency": ["GrossProfitability","Accruals%"],
}
BUCKET_WEIGHTS = {
    "Valuation": 0.25,
    "Quality":   0.20,
    "Growth":    0.20,
    "Momentum":  0.20,
    "Leverage":  0.10,
    "Efficiency":0.05,
}

def winsorize(s: pd.Series, p=0.01) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s
    lo, hi = np.nanpercentile(s, [100*p, 100*(1-p)])
    return s.clip(lo, hi)

def stdize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu, sd = np.nanmean(s), np.nanstd(s)
    return (s - mu) / (sd if (sd and sd > 0) else 1.0)

def industry_neutralize(df: pd.DataFrame, col: str, industry_col="Industry") -> pd.Series:
    if industry_col not in df.columns:
        return df[col]
    out = df[col].copy()
    for _, idx in df.groupby(industry_col).groups.items():
        m = np.nanmean(df.loc[idx, col])
        out.loc[idx] = df.loc[idx, col] - m
    return out

def prepare_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Latest Price" in out and "VWAP" in out:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["Mom_VWAP_Diff%"] = ((out["Latest Price"] - out["VWAP"]) / out["VWAP"]) * 100
    if "Ret12m" in out:
        out["TTM-Return"] = out["Ret12m"]

    invert_cols = set(["P/E Ratio","P/B Ratio","EV/EBITDA","Price/Sales","DebtToEquity"])

    for cols in FACTOR_BUCKETS.values():
        for c in cols:
            if c in out.columns:
                s = winsorize(out[c])
                s = stdize(s)
                if c in invert_cols:
                    s = -s
                out[c] = s

    for cols in FACTOR_BUCKETS.values():
        for c in cols:
            if c in out.columns:
                out[c] = industry_neutralize(out, c, "Industry")

    for b, cols in FACTOR_BUCKETS.items():
        valid = [c for c in cols if c in out.columns]
        if valid:
            out[b] = np.nanmean(np.vstack([out[c] for c in valid]), axis=0)
        else:
            out[b] = np.nan

    def comp_row(r):
        parts = {k: r.get(k) for k in BUCKET_WEIGHTS if k in r and np.isfinite(r.get(k))}
        if not parts:
            return np.nan
        tw = sum(BUCKET_WEIGHTS[k] for k in parts)
        return sum(BUCKET_WEIGHTS[k]*parts[k] for k in parts) / (tw if tw>0 else 1.0)

    out["CompositeScore"] = out.apply(comp_row, axis=1)

    out["RiskFlags"] = ""
    if "P/E Ratio" in df.columns:
        out.loc[pd.to_numeric(df["P/E Ratio"], errors="coerce") <= 0, "RiskFlags"] += " • Negative earnings"
    if "DebtToEquity" in df.columns:
        out.loc[pd.to_numeric(df["DebtToEquity"], errors="coerce") > 3, "RiskFlags"] += " • High leverage"
    if "MarketCap" in df.columns:
        out.loc[pd.to_numeric(df["MarketCap"], errors="coerce") < 1_000, "RiskFlags"] += " • Small-cap/illiquid?"
    if "EPS_TTM" in df.columns:
        out.loc[pd.to_numeric(df["EPS_TTM"], errors="coerce") <= 0, "RiskFlags"] += " • Negative EPS (P/E N/M)"

    return out

def build_waterfall_dict(row: pd.Series):
    parts = {}
    for k in BUCKET_WEIGHTS:
        v = row.get(k)
        if v is not None and np.isfinite(v):
            parts[k] = BUCKET_WEIGHTS[k] * v
    return parts

# -------------------- ORCHESTRATOR (CORE ANALYSIS) --------------------
def analyze_ticker_pro(ticker: str, peer_cap: int = 6):
    """
    Core analysis pipeline:
      - Build peer set
      - Pull metrics
      - Compute factors & composite score
      - Build text + news outputs
    """
    ticker = ticker.upper().strip()
    peers = build_peerset(ticker, vendor_max=20, final_cap=peer_cap) or []

    rows = []
    tgt = get_metrics(ticker)
    if tgt:
        rows.append(tgt)
    for p in peers:
        try:
            rows.append(get_metrics(p))
            time.sleep(0.2)
        except Exception:
            continue

    df = pd.DataFrame(rows)

    if not df.empty and "Ticker" in df.columns:
        prices, vwaps, rets = {}, {}, {}
        for s in df["Ticker"]:
            lp, vw, _ = calculate_vwap(s)
            prices[s] = lp
            vwaps[s] = vw
            rets[s] = get_total_return(s, months=12)
        df["Latest Price"] = df["Ticker"].map(prices)
        df["VWAP"]         = df["Ticker"].map(vwaps)
        df["Ret12m"]       = df["Ticker"].map(rets)

    scored = prepare_factors(df)

    focus_row = None
    if (not scored.empty) and ("Ticker" in scored.columns):
        m = scored["Ticker"] == ticker
        if m.any():
            focus_row = scored.loc[m].iloc[0]

    parts = build_waterfall_dict(focus_row) if focus_row is not None else {}
    _ = parts  # currently unused in UI

    text_synopsis_md = get_company_text_summary(ticker, focus_row if focus_row is not None else {})
    metrics_summary_md = get_company_metrics_summary(focus_row if focus_row is not None else {})

    news_items = get_company_news(ticker, n=8)
    news_md = "### Recent Headlines\n" + render_news_md(news_items)

    focus_row_dict = focus_row.to_dict() if isinstance(focus_row, pd.Series) else None

    return (
        scored,
        text_synopsis_md,
        metrics_summary_md,
        focus_row_dict,
        news_md,
    )

# -------------------- CHARTS --------------------
def charts(scored: pd.DataFrame, focus: str):
    fig1 = None; fig2 = None; fig3 = None
    try:
        if not scored.empty and {"Ticker","CompositeScore"}.issubset(scored.columns):
            dfb = scored[["Ticker","CompositeScore"]].dropna().sort_values("CompositeScore", ascending=False)
            if not dfb.empty:
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                bars = ax1.bar(dfb["Ticker"], dfb["CompositeScore"])
                try:
                    idx = list(dfb["Ticker"]).index(focus)
                    bars[idx].set_linewidth(2.8); bars[idx].set_linestyle("--")
                except ValueError:
                    pass
                ax1.set_title("Composite Score (higher is better)")
                ax1.set_xlabel("Ticker"); ax1.set_ylabel("Composite Score")
    except Exception:
        fig1 = None

    try:
        needed = {"EV/EBITDA","RevenueGrowth%","MarketCap","CompositeScore","Ticker"}
        if not scored.empty and needed.issubset(set(scored.columns)):
            dfx = scored[list(needed)].dropna()
            if not dfx.empty:
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                sizes = np.clip(
                    (pd.to_numeric(dfx["MarketCap"], errors="coerce")
                     / pd.to_numeric(dfx["MarketCap"], errors="coerce").max()) * 800,
                    20, 800
                )
                ax2.scatter(dfx["EV/EBITDA"], dfx["RevenueGrowth%"], s=sizes, alpha=0.0)
                ax2.set_title("EV/EBITDA vs Revenue Growth (size≈MktCap, color=Composite)")
                ax2.set_xlabel("EV/EBITDA"); ax2.set_ylabel("Revenue Growth %")
                if focus in set(dfx["Ticker"]):
                    row = dfx[dfx["Ticker"] == focus].iloc[0]
                    ax2.annotate(focus, (row["EV/EBITDA"], row["RevenueGrowth%"]),
                                 xytext=(5, 5), textcoords="offset points")
    except Exception:
        fig2 = None

    try:
        buckets = ["Valuation","Quality","Growth","Momentum","Leverage","Efficiency"]
        cols = ["Ticker"] + buckets
        if not scored.empty and set(cols).issubset(scored.columns):
            d = scored[cols].set_index("Ticker").dropna(how="all")
            if not d.empty:
                fig3 = plt.figure()
                ax3 = fig3.add_subplot(111)
                im = ax3.imshow(d[buckets].values, aspect="auto")
                ax3.set_xticks(range(len(buckets))); ax3.set_xticklabels(buckets, rotation=0)
                ax3.set_yticks(range(len(d.index))); ax3.set_yticklabels(list(d.index))
                ax3.set_title("Peer Factor Heatmap (bucket z-scores)")
                cb3 = fig3.colorbar(im); cb3.set_label("Z score")
    except Exception:
        fig3 = None

    return fig1, fig2, fig3

HEATMAP_METRIC_DEFS = [
    {"label": "Revenue Growth (TTM)", "value_key": "RevenueGrowth%", "relative_key": "RevenueGrowth%", "fmt": "percent"},
    {"label": "Gross Margin", "value_key": "GrossMargin%", "relative_key": "GrossMargin%", "fmt": "percent"},
    {"label": "FCF Margin", "value_key": "FCF Margin%", "relative_key": "FCF Margin%", "fmt": "percent"},
    {"label": "ROE", "value_key": "ROE%", "relative_key": "ROE%", "fmt": "percent"},
    {"label": "Debt/EBITDA", "value_key": "Debt/EBITDA", "relative_key": "Debt/EBITDA", "fmt": "ratio"},
    {"label": "P/E vs Peers", "value_key": "P/E (raw)", "relative_key": "P/E Ratio", "fmt": "ratio"},
]

def _format_metric_display_value(value, fmt):
    if value is None or not np.isfinite(value):
        return "N/A"
    if fmt == "percent":
        return f"{value:+.1f}%"
    if fmt == "ratio":
        return f"{value:.1f}x"
    if fmt == "float":
        return f"{value:,.2f}"
    return str(value)

def build_metric_heatmap_figure(res: Dict[str, Any]):
    focus = res.get("focus_row") or {}
    base = res.get("base_metrics") or {}
    cells = []
    for metric in HEATMAP_METRIC_DEFS:
        actual = base.get(metric["value_key"])
        relative = focus.get(metric["relative_key"])
        cells.append(
            {
                "label": metric["label"],
                "formatted": _format_metric_display_value(actual, metric["fmt"]),
                "relative": float(relative) if relative is not None and np.isfinite(relative) else 0.0,
            }
        )

    if not cells:
        return None

    cols = 3
    rows = math.ceil(len(cells) / cols)
    z_matrix = []
    text_matrix = []
    for r in range(rows):
        row_z = []
        row_text = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(cells):
                entry = cells[idx]
                row_z.append(entry["relative"])
                row_text.append(f"{entry['label']}<br><span style='font-size:16px;font-weight:600;'>{entry['formatted']}</span>")
            else:
                row_z.append(0.0)
                row_text.append("")
        z_matrix.append(row_z)
        text_matrix.append(row_text)

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix,
            x=[f"Col {i+1}" for i in range(cols)],
            y=[f"Row {i+1}" for i in range(rows)],
            text=text_matrix,
            hoverinfo="text",
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#b91d47"],
                [0.5, "#1f2933"],
                [1.0, "#0f9d58"],
            ],
            zmid=0,
            showscale=False,
            xgap=4,
            ygap=4,
        )
    )
    fig.update_layout(
        margin=dict(l=4, r=4, t=20, b=4),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.4)", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.4)", zeroline=False, showticklabels=False),
        height=260,
    )
    return fig

def five_day_price_plot(ticker: str):
    plt_fig = None
    try:
        _, _, hist = calculate_vwap(ticker.upper())
        if hist is not None and not hist.empty:
            plt_fig = plt.figure()
            ax = plt_fig.add_subplot(111)
            ax.plot(hist.index, hist["Close"])
            ax.set_title(f"{ticker.upper()} — Close (5d)")
            ax.set_xlabel("Date"); ax.set_ylabel("Price")
    except Exception:
        plt_fig = None
    return plt_fig


# -------------------- LOCAL STORAGE HELPERS --------------------
NOTES_FILE = "research_notes.json"
THESES_FILE = "theses.json"

def _load_json(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save {path}: {e}")

# ---------- SESSION SETUP ----------
if "recent_tickers" not in st.session_state:
    st.session_state.recent_tickers = []    # list[{"ticker","time"}]
if "last_results" not in st.session_state:
    st.session_state.last_results = None    # latest analysis bundle
if "notes_store" not in st.session_state:
    st.session_state.notes_store = _load_json(NOTES_FILE, {})
if "theses_store" not in st.session_state:
    st.session_state.theses_store = _load_json(THESES_FILE, [])

# ======================================================================
# ANALYSIS WRAPPER
# ======================================================================
def run_equity_analysis(ticker: str, max_peers: int = 6) -> Dict[str, Any]:
    """
    Wraps analyze_ticker_pro + valuation + charts into a single object
    that the UI can reuse across pages.
    """
    ticker = ticker.upper().strip()

    try:
        (
            scored,
            text_synopsis_md,
            metrics_summary_md,
            focus_row,
            news_md,
        ) = analyze_ticker_pro(ticker, peer_cap=max_peers)
    except Exception as e:
        logging.error(f"analyze_ticker_pro failed for {ticker}: {e}", exc_info=True)
        raise Exception(f"Failed to analyze {ticker}. Ticker may be invalid or data unavailable. (Internal error: {e})")

    overview_md = (
        "### Company Overview\n\n"
        + text_synopsis_md
        + "\n\n"
        + metrics_summary_md
    )

    raw_metrics = None
    if isinstance(scored, pd.DataFrame) and not scored.empty:
        if "Ticker" in scored.columns:
            focus_df = scored[scored["Ticker"] == ticker]
            if not focus_df.empty:
                raw_metrics = focus_df
            else:
                raw_metrics = scored.head(1)
        else:
            raw_metrics = scored.head(1)

    valuation: Dict[str, Dict[str, Any]] = {}
    base_metrics = get_metrics(ticker)
    base_params = _build_scenario_params(base_metrics, "Base")

    for scen in ["Base", "Bull", "Bear"]:
        try:
            df_val, md_val = _scenario_valuation_core(ticker, max_peers, scen)
        except Exception as e:
            df_val, md_val = pd.DataFrame(), f"_Error in {scen} scenario: {e}_"
        valuation[scen] = {"df": df_val, "md": md_val}

    fig_comp, fig_scatter, _ = charts(scored, ticker)
    fig_price = five_day_price_plot(ticker)

    return {
        "ticker": ticker,
        "overview_md": overview_md,
        "focus_row": focus_row,
        "peers_df": scored,
        "valuation": valuation,
        "news_md": news_md,
        "raw_metrics": raw_metrics,
        "charts": {"price": fig_price, "scatter": fig_scatter},
        "base_params": base_params,
        "current_price": get_price_and_shares(ticker)[0],
        "base_metrics": base_metrics,
    }

# ======================================================================
# VALUATION / SCENARIO HELPERS
# ======================================================================
def _estimate_fcff_and_net_debt(symbol: str):
    """Estimates TTM FCFF and latest Net Debt."""
    try:
        t = yf.Ticker(symbol)
        q_cf = t.quarterly_cashflow
        q_bs = t.quarterly_balance_sheet

        cfo_ttm = ttm_from_rows(q_cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex_ttm = ttm_from_rows(q_cf, ["Capital Expenditure", "Change In Property Plant Equipment"])
        
        fcff_ttm = np.nan
        if np.isfinite(cfo_ttm) and np.isfinite(capex_ttm):
            fcff_ttm = cfo_ttm - abs(capex_ttm)
        elif np.isfinite(cfo_ttm):
            fcff_ttm = cfo_ttm

        total_debt_aliases = ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"]
        total_debt = last_q_from_rows(q_bs, total_debt_aliases)
        
        cash_aliases = ["Cash And Cash Equivalents", "Cash", "Cash And Short Term Investments"]
        cash = last_q_from_rows(q_bs, cash_aliases)

        net_debt = np.nan
        if np.isfinite(total_debt) and np.isfinite(cash):
            net_debt = total_debt - cash
        elif np.isfinite(total_debt):
            net_debt = total_debt
        
        return fcff_ttm, net_debt
    except Exception as e:
        logging.warning(f"Failed _estimate_fcff_and_net_debt for {symbol}: {e}")
        return np.nan, np.nan

def _build_scenario_params(metrics, scenario):
    """Creates valuation assumptions based on scenario."""
    base_g = metrics.get("RevenueGrowth%")
    base_m = metrics.get("EBITDAMargin%")

    if not np.isfinite(base_g): base_g = 5.0
    if not np.isfinite(base_m): base_m = 15.0

    params = { "wacc": 0.08, "terminal_g": 0.02, "g_proj": base_g, "m_proj": base_m }

    if scenario == "Bull":
        params["wacc"] = 0.075
        params["terminal_g"] = 0.025
        params["g_proj"] = base_g * 1.2 + 2.0
        params["m_proj"] = base_m + 2.0
    elif scenario == "Bear":
        params["wacc"] = 0.09
        params["terminal_g"] = 0.015
        params["g_proj"] = base_g * 0.8 - 1.0
        params["m_proj"] = base_m - 2.0
    
    return params

def _scenario_valuation_core(ticker: str, max_peers: int, scenario: str):
    """Runs DCF and Comps valuation for a given scenario."""
    price, shares = np.nan, np.nan
    metrics = {}
    try:
        metrics = get_metrics(ticker)
        fcff_ttm, net_debt = _estimate_fcff_and_net_debt(ticker)
        price, shares = get_price_and_shares(ticker)
        
        t_is = yf.Ticker(ticker).quarterly_income_stmt
        rev_ttm = ttm_from_rows(t_is, ["Total Revenue", "Revenue"])
        
        ebitda_ttm = np.nan
        if np.isfinite(metrics.get("EBITDAMargin%")) and np.isfinite(rev_ttm):
            ebitda_ttm = metrics.get("EBITDAMargin%") * rev_ttm / 100.0

        if not np.isfinite(ebitda_ttm):
            ebit_ttm = ttm_from_rows(t_is, ["Ebit","EBIT","Operating Income"])
            dep_ttm = ttm_from_rows(yf.Ticker(ticker).quarterly_cashflow,
                                     ["Depreciation","Depreciation And Amortization"])
            if np.isfinite(ebit_ttm) and np.isfinite(dep_ttm):
                ebitda_ttm = ebit_ttm + dep_ttm

        params = _build_scenario_params(metrics, scenario)
        wacc = params["wacc"]
        term_g = params["terminal_g"]
        g_proj_frac = params["g_proj"] / 100.0

        implied_price_dcf = np.nan
        if all(np.isfinite([fcff_ttm, g_proj_frac, wacc, term_g, net_debt, shares])) and shares > 0 and wacc > term_g:
            try:
                pv_fcffs = []
                last_fcff = fcff_ttm
                for i in range(1, 6):
                    last_fcff = last_fcff * (1 + g_proj_frac)
                    pv_fcffs.append(last_fcff / ((1 + wacc) ** i))
                
                tv = (last_fcff * (1 + term_g)) / (wacc - term_g)
                pv_tv = tv / ((1 + wacc) ** 5)
                
                enterprise_value_dcf = sum(pv_fcffs) + pv_tv
                equity_value_dcf = enterprise_value_dcf - net_debt
                implied_price_dcf = equity_value_dcf / shares
            except Exception as e:
                logging.warning(f"DCF calculation failed for {ticker}: {e}")
                implied_price_dcf = np.nan

        implied_price_ev_ebitda, implied_price_ps = np.nan, np.nan
        try:
            peers = build_peerset(ticker, final_cap=max_peers)
            if peers:
                peer_metrics = [get_metrics(p) for p in peers]
                df_peers = pd.DataFrame(peer_metrics).dropna(subset=["EV/EBITDA (raw)", "P/S (raw)"], how="all")
                
                median_ev_ebitda = np.nanmedian(df_peers["EV/EBITDA (raw)"])
                median_ps = np.nanmedian(df_peers["P/S (raw)"])

                if np.isfinite(median_ev_ebitda) and np.isfinite(ebitda_ttm) and np.isfinite(net_debt) and shares > 0:
                    ev = ebitda_ttm * median_ev_ebitda
                    eq_val = ev - net_debt
                    implied_price_ev_ebitda = eq_val / shares
                
                if np.isfinite(median_ps) and np.isfinite(rev_ttm) and shares > 0:
                    mkt_cap = rev_ttm * median_ps
                    implied_price_ps = mkt_cap / shares
        except Exception as e:
            logging.warning(f"Comps valuation failed for {ticker}: {e}")

        df_data = {
            "Method": ["DCF (5y)", "Comps (EV/EBITDA)", "Comps (P/S)"],
            "Implied Price": [implied_price_dcf, implied_price_ev_ebitda, implied_price_ps]
        }
        df = pd.DataFrame(df_data).dropna(subset=["Implied Price"])
        
        md = f"#### {scenario} Scenario\n**Assumptions:**\n"
        md += f"- Proj. Growth: `{params['g_proj']:.1f}%`\n"
        md += f"- WACC: `{wacc:.1%}`\n"
        md += f"- Terminal Growth: `{term_g:.1%}`\n"
        
        if df.empty:
            md += "\n_Insufficient data for valuation._"
        else:
            if np.isfinite(price):
                df["Premium"] = (df["Implied Price"] / price) - 1.0
                df["Premium"] = df["Premium"].map(lambda x: f"{x:+.1%}")
            df["Implied Price"] = df["Implied Price"].map(lambda x: f"${x:.2f}")

        return df, md
    
    except Exception as e:
        logging.error(f"Failed _scenario_valuation_core for {ticker} ({scenario}): {e}")
        return pd.DataFrame(), f"_Valuation failed for {scenario}: {e}_"

#User interface

def inject_global_css():
    st.markdown(
        """
        <style>
        /* ===== COLOR PALETTE ===== */
        :root {
            --color-primary-bg: #001f3f;    /* Dark Navy Blue */
            --color-secondary-bg: #F5EAAA; /* Khaki */
            --color-page-bg: #FFFFFF;      /* White */
            
            --color-primary-text: #001f3f;    /* Dark Navy Blue */
            --color-secondary-text: #F5EAAA; /* Khaki */
            --color-tertiary-text: #FFFFFF;  /* White */
            
            --color-dark-card-bg: #020617;
            --color-dark-card-text: #E5E7EB;
            --color-dark-card-border: #1F2937;
        }
        
        /* ===== GLOBAL LAYOUT ===== */
        html, body, .stApp {
            background: linear-gradient(90deg, #021026 0%, #021026 22%, #f4f7fb 22%, #ffffff 100%) !important;
            color: var(--color-primary-text) !important;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
            margin: 0 !important;
            padding: 0 !important;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
        }
        div.block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: none !important;
        }
        div[data-testid="stAppViewContainer"] {
            padding-top: 0 !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--color-primary-text) !important;
        }

        /* =================================================
        BUTTON TEXT FIX
        ================================================= */
        button[data-testid="stButton"] {
            color: #FFFFFF !important;
        }
        div[data-testid="stButton"] p {
             color: #FFFFFF !important;
        }

        /* ===== INFO BOX ===== */
        div[data-testid="stInfo"] {
            background-color: #E6F6FF !important; 
            border: 1px solid #B0E0FF !important; 
            color: #001f3f !important;
        }
        div[data-testid="stInfo"] p {
             color: #001f3f !important;
        }
        
        /* ===== PAGE HEADER / HERO ===== */
        .header-hero {
            width: 100%;
            position: relative;
            left: 0;
            transform: none;
            padding: 42px 0 32px 0;
            background: #001a40;
            border-bottom: 2px solid #0f172a;
            box-shadow: 0 15px 35px rgba(15, 23, 42, 0.45);
            z-index: 10;
        }
        .page-header {
            max-width: 1100px;
            margin: 0 auto;
            text-align: center;
        }
        .page-title {
            font-family: 'DM Serif Display', serif;
            font-size: 3rem;
            font-weight: 500;
            color: #ffffff !important;
            margin-bottom: 0.2rem;
            letter-spacing: -0.03em;
        }
        .page-subtitle {
            font-size: 1rem;
            font-weight: 500;
            color: rgba(229, 231, 235, 0.9);
            margin: 0;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .page-mini-desc {
            font-size: 0.9rem;
            color: rgba(229, 231, 235, 0.9);
            margin-top: 0.4rem;
        }

        /* ===== CARD UI ===== */
        .hero-card {
            background: var(--color-primary-bg);
            border-radius: 16px;
            padding: 24px 28px;
            margin-top: 16px;
            margin-bottom: 16px;
            border: 1px solid var(--color-secondary-bg);
        }
        .hero-card .hero-title, .hero-card .hero-subtitle {
             color: var(--color-tertiary-text) !important;
        }
        
        .section-card {
            background: var(--color-page-bg);
            border-radius: 16px;
            padding: 18px 20px;
            margin-top: 12px;
            margin-bottom: 12px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .section-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--color-primary-text);
            margin-bottom: 12px;
        }

        /* ===== METRIC CARD STYLING (VIX, USD, etc.) ===== */
        .metric-card-box {
            background: #f8fafc !important;
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
            height: 100%;
        }
        .metric-label {
            font-size: 13px;
            font-weight: 600;
            color: #001f3f !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
            opacity: 0.85;
        }
        .metric-value-custom {
            color: #001f3f !important;
            font-weight: 700 !important;
            font-size: 26px !important;
            line-height: 1.3;
            margin-bottom: 4px;
        }
        .metric-delta-custom span {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        .metric-delta-custom.positive span {
            color: #047857 !important;
            background-color: #D1FAE5;
        }
        .metric-delta-custom.negative span {
            color: #991B1B !important;
            background-color: #FEE2E2;
        }

        /* ===== TABS FIX ===== */
        button[data-testid="stTab"] {
            color: #4B5563 !important;
            font-weight: 600 !important;
            opacity: 0.7 !important;
        }
        button[data-testid="stTab"][aria-selected="true"] {
            color: var(--color-primary-text) !important;
            opacity: 1 !important;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: var(--color-primary-text) !important; 
        }

        /* ===== TICKER TAPE ===== */
        @keyframes scroll-left {
            from { transform: translateX(0); }
            to   { transform: translateX(-25%); }
        }
        .ticker-tape-container {
            background: #001528;
            color: #f8fafc;
            overflow: hidden;
            padding: 10px 0;
            width: 100%;
            position: relative;
            left: 0;
            margin-left: 0;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .ticker-tape-inner {
            display: inline-flex;
            white-space: nowrap;
            width: max-content;
            animation: scroll-left 40s linear infinite;
        }
        .ticker-item {
            display: inline-block;
            padding: 0 25px;
            font-size: 16px;
            font-weight: 500;
        }
        .ticker-section-label {
            padding: 0 25px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            opacity: 0.75;
            color: var(--color-secondary-text);
        }
        .ticker-symbol {
            font-weight: 700 !important;
            color: var(--color-secondary-text) !important;
            margin-right: 8px;
        }
        .ticker-price {
            margin-right: 8px;
            opacity: 0.9;
        }
        .ticker-change.positive {
            color: #00E500 !important;
            font-weight: 600;
        }
        .ticker-change.negative {
            color: #FF2E2E !important;
            font-weight: 600;
        }

        /* ===== INDEX SNAPSHOT CARDS ===== */
        .index-chart-card {
            background: #f8fafc;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
            display: flex;
            flex-direction: column;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border 0.25s ease;
        }
        .index-chart-title {
            font-size: 1rem;
            font-weight: 700;
            color: var(--color-primary-text);
            margin-bottom: 8px;
        }
        .index-chart-price {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--color-primary-text);
            margin-right: 10px;
            display: block; 
            margin-bottom: 4px;
            line-height: 1.2;
        }
        .index-chart-change {
            font-size: 0.95rem;
            font-weight: 600;
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
        }
        .index-chart-change.positive {
            color: #047857; 
            background-color: #D1FAE5;
        }
        .index-chart-change.negative {
            color: #991B1B; 
            background-color: #FEE2E2;
        }
        .index-metric-list {
            margin-top: 16px;
            border-top: 1px solid #E2E8F0;
            padding-top: 12px;
            flex-grow: 1; 
        }
        .index-metric-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem; 
            color: #374151; 
            margin-bottom: 8px;
            line-height: 1.3;
        }
        .index-metric-label {
            font-weight: 500;
            color: #4B5563;
        }
        .index-metric-value {
            font-weight: 600;
            color: var(--color-primary-text); 
            text-align: right;
            padding-left: 8px; 
        }

        /* ===== VALUATION PAGE INPUTS ===== */
        div[data-testid="stNumberInput"] input {
             color: #001f3f !important;
        }

        /* ======================================================
           SCREENER / ANALYSIS COMMAND CENTER (Google x Apple x JPM)
        =======================================================*/
        .analysis-shell {
            margin-top: 8px;
            margin-bottom: 8px;
        }

        .analysis-gradient {
            background: radial-gradient(circle at top left, #0B1120 0%, #020617 38%, #020617 100%);
            border-radius: 20px;
            padding: 18px 20px 20px 20px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow:
                0 24px 60px rgba(15, 23, 42, 0.45),
                0 0 0 1px rgba(15, 23, 42, 0.5) inset;
        }

        .analysis-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
            gap: 18px;
            align-items: stretch;
        }

        .command-card {
            background: linear-gradient(140deg, #020617 0%, #020617 40%, #020617 100%);
            border-radius: 16px;
            padding: 16px 18px 18px 18px;
            border: 1px solid rgba(148, 163, 184, 0.5);
            color: var(--color-dark-card-text);
            position: relative;
            overflow: hidden;
        }
        .command-card::before {
            content: "";
            position: absolute;
            inset: -80px;
            background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 55%);
            opacity: 0.9;
            pointer-events: none;
        }
        .command-inner {
            position: relative;
            z-index: 1;
        }
        .command-title-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 4px;
        }
        .command-title {
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #E5E7EB;
        }
        .command-pill {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.6);
            color: #E5E7EB;
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.7), rgba(17, 24, 39, 0.9));
        }
        .command-subtitle {
            font-size: 0.85rem;
            color: #9CA3AF;
            margin-bottom: 14px;
        }
        .command-steps {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 10px;
            font-size: 0.78rem;
            color: #9CA3AF;
        }
        .command-step {
            padding: 4px 9px;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.4);
            background: radial-gradient(circle at top left, rgba(148, 163, 184, 0.18), rgba(15, 23, 42, 0.9));
        }
        .command-step.active {
            border-color: #FACC15;
            background: radial-gradient(circle at top left, rgba(250, 204, 21, 0.25), rgba(15, 23, 42, 0.95));
            color: #FEF3C7;
        }
        .command-input-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #9CA3AF;
            margin-bottom: 4px;
        }
        .command-footnote {
            margin-top: 6px;
            font-size: 0.75rem;
            color: #6B7280;
        }

        .module-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }
        .module-card {
            background: radial-gradient(circle at top left, rgba(30, 64, 175, 0.35), rgba(15, 23, 42, 1));
            border-radius: 14px;
            padding: 10px 11px 11px 11px;
            border: 1px solid rgba(148, 163, 184, 0.45);
            color: #E5E7EB;
            font-size: 0.8rem;
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .module-chip-row {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.76rem;
            color: #9CA3AF;
        }
        .module-dot {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: #22C55E;
            box-shadow: 0 0 12px rgba(34, 197, 94, 0.9);
        }
        .module-title {
            font-size: 0.92rem;
            font-weight: 600;
            color: #F9FAFB;
        }
        .module-desc {
            font-size: 0.76rem;
            color: #9CA3AF;
        }
        .module-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-top: 3px;
        }
        .module-tag {
            font-size: 0.7rem;
            padding: 3px 7px;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.4);
            color: #E5E7EB;
            background: rgba(15, 23, 42, 0.9);
        }

        .empty-state-pro {
            margin-top: 14px;
            border-radius: 14px;
            padding: 12px 14px;
            background: rgba(239, 246, 255, 0.96);
            border: 1px dashed rgba(148, 163, 184, 0.8);
            font-size: 0.86rem;
            color: #0F172A;
        }
        .empty-state-title {
            font-weight: 600;
            margin-bottom: 4px;
        }
        .empty-state-list {
            margin: 0;
            padding-left: 1.2rem;
            font-size: 0.85rem;
        }
        .empty-state-list li {
            margin-bottom: 2px;
        }

        /* Smaller screens: stack analysis grid */
        @media (max-width: 900px) {
            .analysis-grid {
                grid-template-columns: minmax(0, 1fr);
            }
        }

        .analysis-hero {
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            background: linear-gradient(120deg, #111827, #1F2937);
            color: #F8FAFC;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
        }
        .analysis-hero h2 {
            margin: 0;
            font-size: 2rem;
            letter-spacing: -0.02em;
        }
        .analysis-hero p {
            margin: 4px 0 0;
            color: rgba(248, 250, 252, 0.78);
        }
        .analysis-hero-meta {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.3em;
            color: rgba(248, 250, 252, 0.65);
        }
        .analysis-empty {
            border-radius: 16px;
            padding: 28px;
            background: rgba(15, 23, 42, 0.08);
            text-align: center;
            border: 1px solid rgba(15, 23, 42, 0.12);
        }
        .analysis-empty-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .metric-card-slim {
            padding: 18px 16px;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(255, 255, 255, 0.8);
            min-height: 90px;
        }
        .metric-card-slim .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            color: #475569;
            margin-bottom: 8px;
        }
        .metric-card-slim .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
        }
        .simplified-overview {
            margin-top: 12px;
            margin-bottom: 18px;
        }
        .market-summary-card {
            background: #F8FAFC;
            border-radius: 16px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
        }
        .market-summary-header {
            font-size: 1.2rem;
            font-weight: 700;
            margin: 0;
            padding: 14px 18px;
            background: linear-gradient(120deg, #0b3d91, #0f172a);
            color: #fff;
            border-radius: 16px 16px 0 0;
            letter-spacing: 0.04em;
        }
        .market-summary-body {
            padding: 20px 24px 28px;
        }
        .market-summary-body p {
            margin-bottom: 12px;
            line-height: 1.5;
            color: #0F172A;
        }
        .mini-card-stack {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .mini-card {
            background: #FFFFFF;
            border-radius: 14px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        }
        .mini-card-header {
            font-size: 1rem;
            font-weight: 600;
            margin: 0;
            padding: 12px 16px;
            background: linear-gradient(120deg, #0b3d91, #0f172a);
            color: #fff;
            border-radius: 14px 14px 0 0;
            letter-spacing: 0.08em;
        }
        .mini-card-content {
            padding: 16px;
        }
        .mini-list {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .mini-list-item {
            font-size: 0.95rem;
            color: #334155;
            display: flex;
            justify-content: space-between;
        }
        .mini-list-item span:first-child {
            font-weight: 600;
            color: #0F172A;
        }
        .mini-card, .metric-card-box, .heatmap-wrapper {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .mini-card:hover, .metric-card-box:hover {
            transform: translateY(-4px);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.15);
        }
        .heatmap-wrapper {
            border-radius: 16px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.12);
            margin-bottom: 12px;
            overflow: hidden;
        }
        /* ===== LEFT SIDE NAV (PRO STYLE) ===== */
        .layout-shell {
            min-height: 100vh;
        }

        .side-nav {
            position: fixed;
            top: 0;
            left: 0;
            width: 220px;
            height: 100vh;
            background: linear-gradient(180deg, #020617 0%, #0b1120 40%, #020617 100%);
            border-radius: 0 32px 32px 0;
            box-shadow: 0 0 40px rgba(15, 23, 42, 0.35);
            padding: 24px 22px 28px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            z-index: 1000;
        }

        .side-nav-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 24px;
        }

        .side-nav-logo {
            width: 32px;
            height: 32px;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 20%, #38bdf8, #0ea5e9 45%, #1d4ed8 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: 700;
            color: #e5f4ff;
        }

        .side-nav-title {
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #94a3b8;
        }

        .side-nav-links {
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-top: 0;
        }

        .side-nav .stButton {
            width: 100%;
        }

        .side-nav .stButton > button {
            width: 100%;
            justify-content: flex-start;
            border-radius: 999px;
            border: none;
            padding: 7px 12px;
            font-size: 13px;
            font-weight: 500;
            background: transparent;
            color: #e5f4ff;
            box-shadow: none;
        }

        .side-nav .stButton > button:hover {
            background: rgba(148, 163, 184, 0.25);
            color: #ffffff;
        }

        /* active state: we’ll add a class through markdown wrapper */
        .side-nav-item-active .stButton > button {
            background: radial-gradient(circle at 0% 0%, #38bdf8 0%, #2563eb 70%);
            color: #ffffff !important;
            box-shadow: 0 0 0 1px rgba(148, 163, 184, 0.25);
        }

        .side-nav-footer {
            font-size: 11px;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #64748b;
            opacity: 0.8;
        }

        /* make main content sit nicely to the right */
        .main-shell {
            width: calc(100vw - 220px);
            margin-left: 220px;
            padding: 0;
            min-height: 100vh;
            box-sizing: border-box;
        }

        .main-content-wrapper {
            width: 100%;
            margin: 0;
            padding: 0;
        }
        .index-chart-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 26px 50px rgba(15, 23, 42, 0.25);
            border-color: rgba(14, 165, 233, 0.4);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_global_header_and_kpis():
    st.markdown(
        """
        <div class="header-hero">
            <div class="page-header">
                <h1 class="page-title">Equity Research Tool</h1>
                <p class="page-subtitle">Fricano Capital Research</p>
                <p class="page-mini-desc">
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Wall Street esque Price Bar ---
def render_dashboard():
    # --- Get data for Ticker Tape AND Index Cards ---
    index_data = get_live_index_data()
    macro_data = get_global_macro_data()

    # ============================
    # ROW 0: Macro Ticker Tape (BLACK BAR DIRECTLY UNDER TITLE)
    # ============================
    ticker_items = macro_data  # Only use macro data for the tape

    if ticker_items:
        item_html_list = []
        item_html_list.append(
            '<div class="ticker-section-label">MACRO DATA</div>'
        )
        for item in ticker_items:
            change_class = "positive" if item["change_val"] >= 0 else "negative"
            item_html_list.append(
                f'<div class="ticker-item">'
                f'  <span class="ticker-symbol">{item["symbol"]}</span>'
                f'  <span class="ticker-price">{item["price_str"]}</span>'
                f'  <span class="ticker-change {change_class}">'
                f'    {item["change_str"]}'
                f'  </span>'
                f'</div>'
            )
        all_items_html = "".join(item_html_list)
        full_ticker_html = f"""
        <div class="ticker-tape-container">
            <div class="ticker-tape-inner">
                {all_items_html}{all_items_html}
            </div>
        </div>
        """
        st.markdown(full_ticker_html, unsafe_allow_html=True)

    # ============================
    # ROW 1: Market Snapshot (TITLE + MACRO CARDS)
    # ============================
    st.markdown("### Market Snapshot")

    # Macro cards (VIX, USD, HY Credit, IG Credit) LIVE INSIDE MARKET SNAPSHOT
    macro_cards = get_macro_indicator_cards()
    if macro_cards:
        macro_cols = st.columns(len(macro_cards))

        for col, card in zip(macro_cols, macro_cards):
            with col:
                # Build the metric in HTML and render once
                val_str = f"{card['value']:,.2f}"
                delta_str = f"{card['change']:+.2f} ({card['pct']:+.2f}%)"
                delta_class = "negative" if card["change"] < 0 else "positive"

                card_html = f"""
                <div class="metric-card-box">
                    <div class="metric-label">{card['label']}</div>
                    <div class="metric-value-custom">{val_str}</div>
                    <div class="metric-delta-custom {delta_class}">
                        <span>{delta_str}</span>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

    st.write("")  # small spacer

    # ============================
    # ROW 2: Index Snapshot Cards (Dow / NASDAQ / S&P / Russell)
    # ============================
    chart_cols = st.columns(4)

    index_map = {
        "Dow Jones": ("^DJI", chart_cols[0]),
        "NASDAQ": ("^IXIC", chart_cols[1]),
        "S&P 500": ("^GSPC", chart_cols[2]),
        "Russell 2000": ("^RUT", chart_cols[3]),
    }

    for display_name, (ticker, col) in index_map.items():
        with col:
            # Build the entire card as one HTML string
            html_parts = ["<div class='index-chart-card'>"]

            # Title always
            html_parts.append(
                f"<div class='index-chart-title'>{display_name}</div>"
            )

            data = get_index_card_metrics(ticker)

            if data is None:
                # Only if *everything* failed (API totally down or bad ticker)
                html_parts.append(
                    "<span class='index-chart-price'>Key metrics unavailable.</span>"
                )
            else:
                last = data["last"]
                chg = data["change"]
                pct = data["change_pct"]

                change_class = "positive" if chg >= 0 else "negative"
                change_sign = "+" if chg >= 0 else ""

                # Price
                html_parts.append(
                    f"<span class='index-chart-price'>${last:,.2f}</span>"
                )

                # Change bar
                html_parts.append(
                    f"<span class='index-chart-change {change_class}'>"
                    f"{change_sign}{chg:,.2f} "
                    f"({change_sign}{pct:.2f}%)"
                    f"</span>"
                )

                # Bottom metrics (YTD, Avg Volume, 52-Wk Range)
                html_parts.append("<div class='index-metric-list'>")
                for metric in data["metrics"]:
                    html_parts.append(
                        f"<div class='index-metric-row'>"
                        f"<span class='index-metric-label'>{metric['label']}</span>"
                        f"<span class='index-metric-value'>{metric['value']}</span>"
                        f"</div>"
                    )
                html_parts.append("</div>")  # close metric list

            html_parts.append("</div>")  # close card
            st.markdown("".join(html_parts), unsafe_allow_html=True)

    st.write("")  # spacer between rows

    # ============================
    # ROW 3: Sector Heatmap
    # ============================
    st.markdown("### Sector Performance")
    sector_perf_data = get_sector_performance()
    if sector_perf_data:
        heatmap_fig = plot_sector_heatmap(sector_perf_data)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.caption(
            "Each tile represents a sector ETF. Color = 1-day % change; tiles sized equally for easier comparison."
        )
    else:
        st.warning("Could not retrieve sector performance data.")

    market_news_items = get_market_news()

    st.markdown("---")

    # ============================
    # ROW 4: Summary + Econ + Smart Money
    # ============================
    left_col, right_col = st.columns([8, 4])

    with left_col:
        summary_text = generate_market_summary(
            index_data, macro_data=macro_data, news_items=market_news_items
        )
        summary_body = "".join(
            f"<p>{part.strip()}</p>" for part in summary_text.strip().split("\n\n") if part.strip()
        )
        st.markdown(
            f"""
            <div class="market-summary-card">
                <div class="market-summary-header">Market Summary (AI-Generated)</div>
                <div class="market-summary-body">
                    {summary_body}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        econ_items = get_economic_calendar()
        smart_items = get_smart_money_signals()
        econ_rows = "".join(
            f"<li class='mini-list-item'><span>{event}</span><span>{time_str}</span></li>"
            for event, time_str in econ_items
        )
        smart_rows = "".join(
            f"<li class='mini-list-item'><span>{label}</span><span>{value}</span></li>"
            for label, value in smart_items.items()
        )
        st.markdown(
            f"""
            <div class="mini-card-stack">
                <div class="mini-card">
                    <div class="mini-card-header">Economic Calendar (Focus)</div>
                    <div class="mini-card-content">
                        <ul class="mini-list">
                            {econ_rows}
                        </ul>
                    </div>
                </div>
                <div class="mini-card">
                    <div class="mini-card-header">Smart Money Tracker</div>
                    <div class="mini-card-content">
                        <ul class="mini-list">
                            {smart_rows}
                        </ul>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # ============================
    # ROW 5: Trending News
    # ============================
    st.markdown(
        "<div class='section-card'><div class='section-title'>Trending Market News</div>",
        unsafe_allow_html=True,
    )
    news_items = market_news_items

    if news_items:
        for n in news_items:
            ts_str = (
                n["time"].strftime("%Y-%m-%d %H:%M")
                if isinstance(n["time"], pd.Timestamp)
                else ""
            )

            # Publisher formatting
            meta = ""
            if n.get("source"):
                meta += f" — {n['source']}"
            if ts_str:
                meta += f" • {ts_str}"

            # Render headline + summary
            st.markdown(
                f"""
                **[{n['headline']}]({n['url']})**{meta}  
                <span style='font-size: 0.9em; color: #666;'>{n['summary'][:240]}...</span>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.write("No recent broad-market headlines available.")

    st.markdown("</div>", unsafe_allow_html=True)

#UX for Analysis Page

def render_watchlist_page():
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Watchlist (coming soon)</div>
            <p>Feature coming soon — keep track of names you monitor most.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_page():
    inject_global_css()

    st.markdown(
        """
        <div class="analysis-hero">
            <div>
                <h2>Equity Research Console</h2>
                <p>A clean, high signal workspace with every stat and context point you need.</p>
            </div>
            <div class="analysis-hero-meta">
                <span>Instant peers · factors · headlines · valuation scaffolding</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_ticker = ""
    if st.session_state.get("last_results"):
        default_ticker = st.session_state.last_results.get("ticker", "")

    ticker = st.text_input(
        "Ticker",
        value=default_ticker,
        key="analysis_page_ticker",
        placeholder="Enter a ticker symbol (e.g., AAPL, MSFT)",
        label_visibility="collapsed",
    ).upper()

    col_univ, col_horz = st.columns(2)
    with col_univ:
        universe = st.selectbox(
            "Universe",
            ["US Large Cap", "US Mid/Small", "Global Developed", "Watchlist"],
            index=0,
            key="screener_universe",
        )
    with col_horz:
        horizon = st.selectbox(
            "Horizon",
            ["3 Months", "12 Months", "3 Years"],
            index=1,
            key="screener_horizon",
        )

    analyze_clicked = st.button("Analyze", use_container_width=True, key="analysis_page_button")

    if analyze_clicked and ticker:
        with st.spinner(f"Analyzing {ticker} — {universe} · {horizon} horizon..."):
            try:
                max_peers = 6
                results = run_equity_analysis(ticker, max_peers=max_peers)
                st.session_state.last_results = results
                st.session_state.recent_tickers.insert(
                    0,
                    {"ticker": results["ticker"], "time": datetime.now().strftime("%Y-%m-%d %H:%M")},
                )
                st.session_state.recent_tickers = st.session_state.recent_tickers[:12]
                st.success(f"{results['ticker']} analysis complete")
                st.session_state.valuation_ticker_input = results["ticker"]
            except Exception as e:
                st.session_state.last_results = None
                logging.error(f"Error during analysis for {ticker}: {e}", exc_info=True)
                st.error(
                    f"Analysis failed for {ticker}. "
                    f"The ticker might be invalid, delisted, or have no data. Error: {e}"
                )

    res = st.session_state.get("last_results")

    if not res:
        st.markdown(
            """
            <div class="analysis-empty">
                <div class="analysis-empty-title">Ready when you are.</div>
                <p>Run a ticker to unlock peer comparisons, full-factor summaries, and the new heatmap view.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    base_metrics = res.get("base_metrics") or {}
    focus_metrics = res.get("focus_row") or {}
    current_price = res.get("current_price")

    summary_items = [
        {"label": "Current Price", "value": current_price, "fmt": "currency"},
        {"label": "Market Cap", "value": base_metrics.get("MarketCap"), "fmt": "marketcap"},
        {"label": "Revenue Growth", "value": base_metrics.get("RevenueGrowth%"), "fmt": "percent", "relative": focus_metrics.get("RevenueGrowth%")},
        {"label": "Gross Margin", "value": base_metrics.get("GrossMargin%"), "fmt": "percent", "relative": focus_metrics.get("GrossMargin%")},
        {"label": "EV/EBITDA", "value": base_metrics.get("EV/EBITDA (raw)"), "fmt": "ratio", "relative": focus_metrics.get("EV/EBITDA")},
        {"label": "Debt/Equity", "value": base_metrics.get("DebtToEquity"), "fmt": "ratio", "relative": focus_metrics.get("DebtToEquity")},
        {"label": "P/E (raw)", "value": base_metrics.get("P/E (raw)"), "fmt": "ratio", "relative": focus_metrics.get("P/E Ratio")},
        {"label": "FCF Margin", "value": base_metrics.get("FCF Margin%"), "fmt": "percent", "relative": focus_metrics.get("FCF Margin%")},
    ]

    def format_summary_value(value, fmt):
        if value is None or not np.isfinite(value):
            return "N/A"
        if fmt == "currency":
            return f"${value:,.2f}"
        if fmt == "marketcap":
            if value >= 1e9:
                return f"${value/1e9:,.2f}B"
            if value >= 1e6:
                return f"${value/1e6:,.1f}M"
            return f"${value:,.0f}"
        if fmt == "percent":
            return f"{value:+.1f}%"
        if fmt == "ratio":
            return f"{value:.1f}x"
        return str(value)

    for i in range(0, len(summary_items), 4):
        batch = summary_items[i : i + 4]
        cols = st.columns(len(batch))
        for col, item in zip(cols, batch):
            with col:
                relative = item.get("relative")
                color = "#111"
                if relative is not None and np.isfinite(relative):
                    color = "#047857" if relative >= 0 else "#dc2626"
                st.markdown(
                    f"""
                    <div class="metric-card-slim">
                        <div class="metric-label">{item['label']}</div>
                        <div class="metric-value" style="color:{color};">
                            {format_summary_value(item["value"], item["fmt"])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
        <div class="section-card simplified-overview">
            <div class="section-title">Summary Narrative</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(res.get("overview_md", ""), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    heatmap_fig = build_metric_heatmap_figure(res)
    if heatmap_fig:
        heatmap_wrapper = "<div class='heatmap-wrapper'>"
    if heatmap_fig:
        st.markdown(
            f"""
            <div class="section-card simplified-overview">
                <div class="section-title">Metric Heatmap</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='heatmap-wrapper'>", unsafe_allow_html=True)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Colored tiles compare your name versus peers. Green = strength, red = relative weakness.")

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Deep Dive Workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_peers, tab_news = st.tabs(["Overview", "Peer Table", "Headlines"])

    with tab_overview:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Company Overview</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(res.get("overview_md", ""), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_peers:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Peer Table</div>
            """,
            unsafe_allow_html=True,
        )

        peers_df = res.get("peers_df")

        if peers_df is not None and not peers_df.empty:
            def highlight_negatives(v):
                try:
                    v_float = float(v)
                except Exception:
                    return ""
                return "color: #ff4d4f;" if v_float < 0 else ""

            styled_peers = peers_df.style.applymap(highlight_negatives)
            st.dataframe(styled_peers, use_container_width=True)
        else:
            st.write("No peer data available for this name.")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_news:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Recent Headlines</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(res.get("news_md", ""), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- VALUATION PAGE ----------
def render_valuation_page():
    # --- MODIFIED --- Added CSS call
    inject_global_css()
    
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Valuation Modeling</div>
            <div class="hero-subtitle">
                Build DCF and multiples-based valuations with scenario analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    def format_currency(x: float) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x))):
            return "N/A"
        return f"${x:,.2f}"

    def format_percent(x: float) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x))):
            return "N/A"
        return f"{x:.1f}%"

    def compute_dcf_price(
        eps: float,
        growth_rate_pct: float,
        discount_rate_pct: float,
        terminal_growth_pct: float,
        years: int = 5,
    ) -> float:
        if eps is None or not np.isfinite(eps) or eps <= 0:
            return np.nan

        g = growth_rate_pct / 100.0
        r = discount_rate_pct / 100.0
        gt = terminal_growth_pct / 100.0

        if r <= gt:
            return np.nan

        pv_flows = []
        cf_t = eps
        for t in range(1, years + 1):
            cf_t = cf_t * (1 + g)
            pv_flows.append(cf_t / ((1 + r) ** t))

        cf_n = cf_t
        cf_n_plus_1 = cf_n * (1 + gt)
        terminal_value_n = cf_n_plus_1 / (r - gt)
        pv_terminal = terminal_value_n / ((1 + r) ** years)

        return sum(pv_flows) + pv_terminal

    def build_sensitivity_table(
        eps: float,
        growth_rates: list,
        discount_rates: list,
        terminal_growth_pct: float,
        years: int = 5,
    ) -> pd.DataFrame:
        data = {}
        for dr in discount_rates:
            row_vals = []
            for gr in growth_rates:
                price = compute_dcf_price(
                    eps=eps,
                    growth_rate_pct=gr,
                    discount_rate_pct=dr,
                    terminal_growth_pct=terminal_growth_pct,
                    years=years,
                )
                row_vals.append(price)
            data[f"{dr}%"] = row_vals
        df = pd.DataFrame(data, index=[f"{g}%" for g in growth_rates])
        return df

    def compute_upside(target: float, current: float) -> float:
        if target is None or not np.isfinite(target) or not np.isfinite(current) or current <= 0:
            return np.nan
        return (target - current) / current * 100.0

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Select Company</div>
            <div class="section-subtitle">Choose a company to value</div>
        """,
        unsafe_allow_html=True,
    )

    default_ticker = ""
    if "valuation_ticker" in st.session_state:
        default_ticker = st.session_state.valuation_ticker
    elif st.session_state.get("last_results"):
        default_ticker = st.session_state.last_results.get("ticker", "")

    st.markdown('<div class="valuation-load-section">', unsafe_allow_html=True)
    
    selected_ticker = st.text_input(
        "Ticker symbol",
        value=default_ticker,
        key="valuation_ticker_input",
        placeholder="Enter ticker symbol (e.g., AAPL, MSFT)",
        label_visibility="collapsed",
    ).upper()
    
    load_clicked = st.button("Load Company", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if load_clicked and selected_ticker:
        tkr = selected_ticker.upper().strip()
        with st.spinner(f"Loading data for {tkr}..."):
            try:
                metrics = get_metrics(tkr)
                price, _ = get_price_and_shares(tkr)
                st.session_state.valuation_ticker = tkr
                st.session_state.valuation_metrics = metrics
                st.session_state.valuation_price = price
                st.success(f"Loaded metrics for {tkr}")
                
                # --- NEW: Trigger analysis run ---
                # Also run a full analysis and save it
                max_peers = 6
                results = run_equity_analysis(tkr, max_peers=max_peers)
                st.session_state.last_results = results
                st.session_state.recent_tickers.insert(
                    0,
                    {"ticker": results["ticker"], "time": datetime.now().strftime("%Y-%m-%d %H:%M")},
                )
                st.session_state.recent_tickers = st.session_state.recent_tickers[:12]
                st.success(f"Full analysis report updated for {tkr}")
                
            except Exception as e:
                logging.error(f"Failed to load metrics for {selected_ticker}: {e}")
                st.error(f"Could not load data for {selected_ticker.upper()}.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    metrics = st.session_state.get("valuation_metrics")
    current_ticker = st.session_state.get("valuation_ticker")
    current_price = st.session_state.get("valuation_price")

    if metrics and current_ticker:
        prof = get_profile(current_ticker) or {}
        company_name = prof.get("name") or current_ticker
        eps = metrics.get("EPS_TTM")
        
        st.markdown(
            """
            <div class="section-card snapshot-card">
                <div class="snapshot-title" style="color: #001f3f !important;">Company Snapshot</div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div style="color: #001f3f !important;">
                <div style="font-size: 0.875rem; color: #001f3f !important;">Ticker</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #001f3f !important;">{current_ticker}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="color: #001f3f !important;">
                <div style="font-size: 0.875rem; color: #001f3f !important;">Company</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #001f3f !important;">{company_name}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div style="color: #001f3f !important;">
                <div style="font-size: 0.875rem; color: #001f3f !important;">Current Price</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #001f3f !important;">{format_currency(current_price or np.nan)}</div>
            </div>
            """, unsafe_allow_html=True)

        if metrics.get("MarketCap"):
            mc = metrics["MarketCap"]
            if np.isfinite(mc):
                st.markdown(f"""
                <div style="font-size: 0.875rem; color: #001f3f !important;">
                    Approx. market cap: ${mc:,.0f}M
                </div>
                """, unsafe_allow_html=True)

        if eps is None or not np.isfinite(eps):
            st.warning(
                "EPS_TTM is not available. DCF andP/E valuations may be limited or N/A."
            )

        st.markdown("</div>", unsafe_allow_html=True) # Closes section-card
        st.write("")

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title" style="color: #001f3f !important;">Scenario Selection</div>
                <div class="section-subtitle" style="color: #001f3f !important;">Choose valuation scenario assumptions</div>
            """,
            unsafe_allow_html=True,
        )

        scenario_presets = {
            "Bull Case": {"revenue_growth": 25.0, "discount_rate": 8.0, "terminal_growth": 4.0, "pe_multiple": 35.0},
            "Base Case": {"revenue_growth": 15.0, "discount_rate": 10.0, "terminal_growth": 3.0, "pe_multiple": 25.0},
            "Bear Case": {"revenue_growth": 5.0,  "discount_rate": 12.0, "terminal_growth": 2.0, "pe_multiple": 15.0},
        }
        
        # 1. Initialize the scenario in session state if it's not there
        if "valuation_scenario" not in st.session_state:
            st.session_state.valuation_scenario = "Base Case"

        # 2. Create columns for the buttons
        scen_c1, scen_c2, scen_c3 = st.columns(3)
        
        # 3. Create the buttons. When clicked, they update session state.
        with scen_c1:
            if st.button("Bull Case", use_container_width=True, key="scen_bull"):
                st.session_state.valuation_scenario = "Bull Case"
        with scen_c2:
            if st.button("Base Case", use_container_width=True, key="scen_base"):
                st.session_state.valuation_scenario = "Base Case"
        with scen_c3:
            if st.button("Bear Case", use_container_width=True, key="scen_bear"):
                st.session_state.valuation_scenario = "Bear Case"
        
        # 4. Read the current scenario from session state
        selected_scenario = st.session_state.valuation_scenario
        
        # 5. (Optional) Show which one is active
        st.markdown(f"**Active Scenario:** `{selected_scenario}`")
        

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Valuation Assumptions</div>
                <div class="section-subtitle">Adjust key assumptions for your valuation model</div>
            """,
            unsafe_allow_html=True,
        )

        preset = scenario_presets[selected_scenario]
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.markdown(f"""
            <p style="font-size: 0.875rem; color: #001f3f !important; margin-bottom: 0.25rem;">
                Revenue / EPS Growth (%)
            </p>
            """, unsafe_allow_html=True)
            revenue_growth = st.number_input(
                "Revenue / EPS Growth (%)",
                value=float(preset["revenue_growth"]),
                step=1.0,
                format="%.1f",
                key="val_rev_growth",
                label_visibility="collapsed" # Hide the default label
            )
        with col_b:
            st.markdown(f"""
            <p style="font-size: 0.875rem; color: #001f3f !important; margin-bottom: 0.25rem;">
                Discount Rate / WACC (%)
            </p>
            """, unsafe_allow_html=True)
            discount_rate = st.number_input(
                "Discount Rate / WACC (%)",
                # --- THIS IS THE FIX ---
                value=float(preset["discount_rate"]), # Changed "discount_state" to "discount_rate"
                # --- END OF FIX ---
                step=0.5,
                format="%.1f",
                key="val_discount_rate",
                label_visibility="collapsed" # Hide the default label
            )
        with col_c:
            st.markdown(f"""
            <p style="font-size: 0.875rem; color: #001f3f !important; margin-bottom: 0.25rem;">
                Terminal Growth Rate (%)
            </p>
            """, unsafe_allow_html=True)
            terminal_growth = st.number_input(
                "Terminal Growth Rate (%)",
                value=float(preset["terminal_growth"]),
                step=0.5,
                format="%.1f",
                key="val_terminal_growth",
                label_visibility="collapsed" # Hide the default label
            )
        with col_d:
            st.markdown(f"""
            <p style="font-size: 0.875rem; color: #001f3f !important; margin-bottom: 0.25rem;">
                Target P/E Multiple
            </p>
            """, unsafe_allow_html=True)
            pe_multiple = st.number_input(
                "Target P/E Multiple",
                value=float(preset["pe_multiple"]),
                step=1.0,
                format="%.1f",
                key="val_pe_multiple",
                label_visibility="collapsed" # Hide the default label
            )

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        if st.button("Run Valuation", use_container_width=True, key="run_valuation_btn"):
            with st.spinner(f"Running valuation for {current_ticker} ({selected_scenario})..."):
                dcf_price = compute_dcf_price(
                    eps=eps if eps is not None else np.nan,
                    growth_rate_pct=revenue_growth,
                    discount_rate_pct=discount_rate,
                    terminal_growth_pct=terminal_growth,
                    years=5,
                )

                if eps is not None and np.isfinite(eps) and eps > 0:
                    pe_val = eps * pe_multiple
                else:
                    pe_val = np.nan

                ps_val = np.nan
                ps_raw = metrics.get("P/S (raw)")
                if (
                    ps_raw is not None
                    and np.isfinite(ps_raw)
                    and ps_raw > 0
                    and metrics.get("MarketCap")
                    and np.isfinite(metrics["MarketCap"])
                ):
                    if current_price and np.isfinite(current_price):
                        target_ps = ps_raw * (pe_multiple / preset["pe_multiple"])
                        ps_val = current_price * (target_ps / ps_raw)

                dcf_upside = compute_upside(dcf_price, current_price)
                pe_upside = compute_upside(pe_val, current_price)
                ps_upside = compute_upside(ps_val, current_price)

                st.markdown(
                    """
                    <div class="section-card">
                        <div class="section-title">Valuation Results</div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "<div class='valuation-metric-label'>DCF Valuation (5-year, terminal at year 5)</div>",
                    unsafe_allow_html=True,
                )
                if np.isfinite(dcf_price):
                    cls = "positive" if (dcf_upside is not None and dcf_upside >= 0) else "negative"
                    st.markdown(
                        f"<div class='valuation-metric-value'>{format_currency(dcf_price)}</div>"
                        f"<div class='valuation-current-price'>Current Price"
                        f"<span style='float:right'>{format_currency(current_price or np.nan)}</span></div>"
                        f"<div class='valuation-upside {cls}'>Upside/Downside"
                        f"<span style='float:right'>{format_percent(dcf_upside)}</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div class='valuation-metric-value'>N/A</div>", unsafe_allow_html=True)

                st.markdown("<hr style='border-top: 1px solid #001f3f;'>", unsafe_allow_html=True)

                st.markdown(
                    "<div class='valuation-metric-label'>P/E-Based Valuation</div>",
                    unsafe_allow_html=True,
                )
                if np.isfinite(pe_val):
                    cls = "positive" if (pe_upside is not None and pe_upside >= 0) else "negative"
                    st.markdown(
                        f"<div class='valuation-metric-value'>{format_currency(pe_val)}</div>"
                        f"<div class='valuation-current-price'>Current Price"
                        f"<span style='float:right'>{format_currency(current_price or np.nan)}</span></div>"
                        f"<div class='valuation-upside {cls}'>Upside/Downside"
                        f"<span style='float:right'>{format_percent(pe_upside)}</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div class'valuation-metric-value'>N/A</div>", unsafe_allow_html=True)

                st.markdown("<hr style='border-top: 1px solid #001f3f;'>", unsafe_allow_html=True)

                st.markdown(
                    "<div class='valuation-metric-label'>P/S-Based Valuation (scaled vs current P/S)</div>",
                    unsafe_allow_html=True,
                )
                if np.isfinite(ps_val):
                    cls = "positive" if (ps_upside is not None and ps_upside >= 0) else "negative"
                    st.markdown(
                        f"<div class='valuation-metric-value'>{format_currency(ps_val)}</div>"
                        f"<div class='valuation-current-price'>Current Price"
                        f"<span style='float:right'>{format_currency(current_price or np.nan)}</span></div>"
                        f"<div class='valuation-upside {cls}'>Upside/Downside"
                        f"<span style'float:right'>{format_percent(ps_upside)}</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div class='valuation-metric-value'>N/A</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
                st.write("")

                st.markdown(
                    """
                    <div class="section-card">
                        <div class="section-title">DCF Sensitivity Analysis</div>
                        <div class="section-subtitle">
                            Price targets across different growth and discount rate assumptions
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                if eps is not None and np.isfinite(eps) and eps > 0:
                    growth_rates = [5, 10, 15, 20, 25]
                    discount_rates = [8, 9, 10, 11, 12]
                    sens_df = build_sensitivity_table(
                        eps=eps,
                        growth_rates=growth_rates,
                        discount_rates=discount_rates,
                        terminal_growth_pct=terminal_growth,
                        years=5,
                    )
                    sens_display = sens_df.applymap(format_currency)
                    st.dataframe(sens_display, use_container_width=True)
                    st.caption(
                        "Rows = growth rates; columns = discount rates; "
                        "values = DCF-implied price per share."
                    )
                else:
                    st.info(
                        "Sensitivity table requires a positive EPS_TTM value. "
                        "Try another ticker with available EPS."
                    )

                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Load a company first, then run valuation.")

# ---------- RESEARCH PAGE ----------
def render_research_page():
    # --- MODIFIED --- Added CSS call
    inject_global_css()
    
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Research Library</div>
            <div class="hero-subtitle">Create and manage private research notes for your tracked companies.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    notes_store: Dict[str, str] = st.session_state.notes_store
    current_from_recent = (
        st.session_state.recent_tickers[0]["ticker"]
        if st.session_state.recent_tickers else ""
    )

    tickers = sorted(
        set(list(notes_store.keys()) + ([current_from_recent] if current_from_recent else []))
    )
    if not tickers:
        tickers = ["(no ticker yet)"]

    selected = st.selectbox("Select ticker for notes", options=tickers)
    key = selected if selected != "(no ticker yet)" else current_from_recent or ""

    existing = notes_store.get(key, "") if key else ""
    
    st.markdown(
        """
        <div class="section-card">
        """,
        unsafe_allow_html=True
    )
    
    new_text = st.text_area("Notes", existing, height=260, key=f"notes_area_{key or 'global'}")

    col_save, col_info = st.columns([1, 3])
    with col_save:
        if st.button("Save Notes"):
            if key:
                st.session_state.notes_store[key] = new_text
                _save_json(NOTES_FILE, st.session_state.notes_store)
                st.success(f"Notes saved for {key}.")
            else:
                st.warning("No ticker selected.")

    with col_info:
        st.caption("Notes are stored locally in `research_notes.json` in this app's directory.")

    if st.session_state.notes_store:
        st.markdown("#### Tickers with saved notes")
        st.write(", ".join(sorted(st.session_state.notes_store.keys())))
        
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- THESES PAGE ----------
def render_theses_page():
    # --- MODIFIED --- Added CSS call
    inject_global_css()
    
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Investment Theses</div>
            <div class="hero-subtitle">Draft and save structured investment theses.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Draft a New Thesis</div>
        """,
        unsafe_allow_html=True
    )

    current_from_recent = (
        st.session_state.recent_tickers[0]["ticker"]
        if st.session_state.recent_tickers else ""
    )

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker", value=current_from_recent, key="thesis_ticker").upper()
        rating = st.selectbox("Rating", ["Buy", "Hold", "Sell"], key="thesis_rating")
        horizon = st.text_input("Time Horizon", value="12–18 months", key="thesis_horizon")
    with col2:
        val_anchor = st.text_area("Valuation Anchor", key="thesis_val", height=80, placeholder="e.g., $150 PT (12x EV/EBITDA)")
    drivers = st.text_area("Key Drivers", key="thesis_drivers", height=120, placeholder="- Driver 1...\n- Driver 2...")
    risks = st.text_area("Risks", key="thesis_risks", height=100, placeholder="- Risk 1...\n- Risk 2...")

    if st.button("Save Thesis"):
        if ticker:
            thesis = {
                "ticker": ticker.upper(),
                "rating": rating,
                "horizon": horizon,
                "drivers": drivers,
                "risks": risks,
                "valuation_anchor": val_anchor,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            st.session_state.theses_store.append(thesis)
            _save_json(THESES_FILE, st.session_state.theses_store)
            st.success(f"Thesis saved for {ticker.upper()}.")
        else:
            st.warning("Please enter a ticker for the thesis.")
            
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    if st.session_state.theses_store:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Saved Theses</div>
            """,
            unsafe_allow_html=True
        )
        df_theses = pd.DataFrame(st.session_state.theses_store)
        st.dataframe(df_theses, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================================
# MAIN APP
# ======================================================================
# ======================================================================
# MAIN APP
# ======================================================================
def main():
    st.set_page_config(
        page_title="Equity Research Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Load DM Serif Display for the title
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:wght@400&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True,
    )

    inject_global_css()

    if "top_nav_page" not in st.session_state:
        st.session_state.top_nav_page = "Dashboard"

    st.markdown("<div class='layout-shell'>", unsafe_allow_html=True)

    st.markdown("<div class='side-nav'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="side-nav-header">
            <div class="side-nav-logo">F</div>
            <div class="side-nav-title">Navigation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='side-nav-links'>", unsafe_allow_html=True)

    def nav_item(label: str, key: str, page_name: str):
        active = st.session_state.top_nav_page == page_name
        wrapper_class = "side-nav-item-active" if active else ""
        class_attr = f" class='{wrapper_class}'" if wrapper_class else ""
        st.markdown(f"<div{class_attr}>", unsafe_allow_html=True)
        if st.button(label, key=key):
            st.session_state.top_nav_page = page_name
        st.markdown("</div>", unsafe_allow_html=True)

    nav_item("Dashboard", "nav_dashboard", "Dashboard")
    nav_item("Screener", "nav_screener", "Analysis")
    nav_item("Valuation", "nav_valuation", "Valuation")
    nav_item("Research", "nav_research", "Research")
    nav_item("Theses", "nav_theses", "Theses")
    nav_item("Watchlist", "nav_watchlist", "Watchlist")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='side-nav-footer'>FRICANO CAPITAL RESEARCH</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='main-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='main-content-wrapper'>", unsafe_allow_html=True)

    page = st.session_state.top_nav_page
    if page == "Dashboard":
        render_global_header_and_kpis()
        render_dashboard()
    elif page == "Analysis":
        render_analysis_page()
    elif page == "Research":
        render_research_page()
    elif page == "Valuation":
        render_valuation_page()
    elif page == "Theses":
        render_theses_page()
    elif page == "Watchlist":
        render_watchlist_page()

    st.markdown("</div></div></div>", unsafe_allow_html=True)


# This block MUST be at the end of the file
# and have no indentation to run the app.
if __name__ == "__main__":
    main()
