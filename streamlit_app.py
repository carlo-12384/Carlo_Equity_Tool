# -*- coding: utf-8 -*-
"""Carlo Equity Tool — Streamlit App (Blocks-style UI)"""

import os, time, math, logging, textwrap, datetime as dt
import requests, pandas as pd, numpy as np, yfinance as yf
import matplotlib.pyplot as plt
from functools import lru_cache
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st

# from scipy.stats import norm # REMOVED: This caused the ModuleNotFoundError

# -------------------- CONFIG / LOGGING --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Note: Removed an invalid character from the end of your key string
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "d49tfm9r01qlaebj5lkgd49tfm9r01qlaebj5ll0") # peers/profile/news
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
        src = f" — {it.get('source')}" if it.get("source") else ""
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

        prof = get_profile(symbol) or {}
        industry = prof.get("finnhubIndustry") or ""

        out.update({
            "MarketCap": mktcap, "Latest Price": price, "EPS_TTM": eps_ttm,
            "P/E (raw)": pe_raw, "P/B (raw)": pb_raw, "EV/EBITDA (raw)": ev_ebitda_raw, "P/S (raw)": ps_raw,
            "P/E Ratio": pe_raw, "P/B Ratio": pb_raw, "EV/EBITDA": ev_ebitda_raw, "Price/Sales": ps_raw,
            "ROE%": roe_pct, "GrossMargin%": gross_margin_pct, "EBITDAMargin%": ebitda_margin_pct,
            "RevenueGrowth%": rev_growth_pct, "DebtToEquity": dte,
            "GrossProfitability": gross_profitability, "AssetGrowth%": asset_growth_pct,
            "Accruals%": accruals_pct, "InterestCoverage": interest_coverage,
            "Industry": industry
        })
    except Exception as e:
        logging.warning(f"get_metrics failed for {symbol}: {e}")
    return out

# -------------------- FACTOR PIPELINE --------------------
FACTOR_BUCKETS = {
    "Valuation": ["P/E Ratio","P/B Ratio","EV/EBITDA","Price/Sales"],
    "Quality":   ["ROE%","GrossMargin%","EBITDAMargin%","InterestCoverage"],
    "Growth":    ["RevenueGrowth%","AssetGrowth%"],
    "Momentum":  ["TTM-Return","Mom_VWAP_Diff%"],
    "Leverage":  ["DebtToEquity"],
    "Efficiency":["GrossProfitability","Accruals%"]
}
BUCKET_WEIGHTS = {"Valuation":0.25,"Quality":0.20,"Growth":0.20,"Momentum":0.20,"Leverage":0.10,"Efficiency":0.05}

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

# --- REMOVED HELPER ---
# def _z_to_percentile(z_score: float) -> float: ...

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

    # --- MODIFICATION: Removed percentile calculation ---
    # factor_percentiles = {}
    # ...
    # --- END MODIFICATION ---

    parts = build_waterfall_dict(focus_row) if focus_row is not None else {}
    peers_used = [p for p in peers if p in list(scored.get("Ticker", []))] if not scored.empty else []

    text_synopsis_md = get_company_text_summary(ticker, focus_row if focus_row is not None else {})
    metrics_summary_md = get_company_metrics_summary(focus_row if focus_row is not None else {})

    # --- REMOVED kpi_ratings_md and waterfall_md generation ---
    # We will build this manually in the UI layer now

    news_items = get_company_news(ticker, n=8)
    news_md = "### Recent Headlines\n" + render_news_md(news_items)

    # --- NEW: Convert focus_row to dict for stable storage ---
    focus_row_dict = focus_row.to_dict() if isinstance(focus_row, pd.Series) else None

    return (
        scored,
        text_synopsis_md,
        metrics_summary_md,
        # factor_percentiles, # REMOVED
        focus_row_dict,          # MODIFIED: Return dict
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
                sc_ = ax2.scatter(dfx["EV/EBITDA"], dfx["RevenueGrowth%"], s=sizes,
                                    c=dfx["CompositeScore"], alpha=0.0) # Changed alpha to 0 for invisible points
                ax2.set_title("EV/EBITDA vs Revenue Growth (size≈MktCap, color=Composite)")
                ax2.set_xlabel("EV/EBITDA"); ax2.set_ylabel("Revenue Growth %")
                if focus in set(dfx["Ticker"]):
                    row = dfx[dfx["Ticker"] == focus].iloc[0]
                    ax2.annotate(focus, (row["EV/EBITDA"], row["RevenueGrowth%"]),
                                 xytext=(5, 5), textcoords="offset points")
                # Removed colorbar as points are invisible
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

# -------------------- THESIS / LIBRARY HELPERS --------------------
# (unchanged from your version; omitted here for brevity if you like,
# but you can keep them exactly as in your original file)
# NOTE: they are not used by the new Blocks UI yet, so they are optional.

# -------------------- VALUATION / SCENARIO HELPERS --------------------
# *** IMPLEMENTED MISSING FUNCTIONS ***

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
            fcff_ttm = cfo_ttm - abs(capex_ttm) # Capex is often negative
        elif np.isfinite(cfo_ttm):
            fcff_ttm = cfo_ttm # Fallback to CFO if capex is missing

        total_debt_aliases = ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"]
        total_debt = last_q_from_rows(q_bs, total_debt_aliases)
        
        cash_aliases = ["Cash And Cash Equivalents", "Cash", "Cash And Short Term Investments"]
        cash = last_q_from_rows(q_bs, cash_aliases)

        net_debt = np.nan
        if np.isfinite(total_debt) and np.isfinite(cash):
            net_debt = total_debt - cash
        elif np.isfinite(total_debt):
            net_debt = total_debt # Fallback if cash is missing
        
        return fcff_ttm, net_debt
    except Exception as e:
        logging.warning(f"Failed _estimate_fcff_and_net_debt for {symbol}: {e}")
        return np.nan, np.nan

def _build_scenario_params(metrics, scenario):
    """Creates valuation assumptions based on scenario."""
    base_g = metrics.get("RevenueGrowth%")
    base_m = metrics.get("EBITDAMargin%")

    # Fallbacks for missing data
    if not np.isfinite(base_g): base_g = 5.0
    if not np.isfinite(base_m): base_m = 15.0

    params = { "wacc": 0.08, "terminal_g": 0.02, "g_proj": base_g, "m_proj": base_m }

    if scenario == "Bull":
        params["wacc"] = 0.075
        params["terminal_g"] = 0.025
        params["g_proj"] = base_g * 1.2 + 2.0  # 20% faster + 200bps
        params["m_proj"] = base_m + 2.0        # 200bps margin expansion
    elif scenario == "Bear":
        params["wacc"] = 0.09
        params["terminal_g"] = 0.015
        params["g_proj"] = base_g * 0.8 - 1.0  # 20% slower - 100bps
        params["m_proj"] = base_m - 2.0        # 200bps margin compression
    
    return params

def _scenario_valuation_core(ticker: str, max_peers: int, scenario: str):
    """Runs DCF and Comps valuation for a given scenario."""
    price, shares = np.nan, np.nan
    metrics = {}
    try:
        # 1. Get Base Data
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
             dep_ttm = ttm_from_rows(yf.Ticker(ticker).quarterly_cashflow, ["Depreciation","Depreciation And Amortization"])
             if np.isfinite(ebit_ttm) and np.isfinite(dep_ttm):
                 ebitda_ttm = ebit_ttm + dep_ttm

        # 2. Get Scenario Params
        params = _build_scenario_params(metrics, scenario)
        wacc = params["wacc"]
        term_g = params["terminal_g"]
        g_proj_frac = params["g_proj"] / 100.0

        # 3. DCF Valuation
        implied_price_dcf = np.nan
        if all(np.isfinite([fcff_ttm, g_proj_frac, wacc, term_g, net_debt, shares])) and shares > 0 and wacc > term_g:
            try:
                pv_fcffs = []
                last_fcff = fcff_ttm
                for i in range(1, 6): # 5-year projection
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


        # 4. Comps Valuation
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

        # 5. Format Output
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
        elif np.isfinite(price):
            df["Premium"] = (df["Implied Price"] / price) - 1.0
            df["Premium"] = df["Premium"].map(lambda x: f"{x:+.1%}")
        
        df["Implied Price"] = df["Implied Price"].map(lambda x: f"${x:.2f}")

        return df, md
    
    except Exception as e:
        logging.error(f"Failed _scenario_valuation_core for {ticker} ({scenario}): {e}")
        return pd.DataFrame(), f"_Valuation failed for {scenario}: {e}_"


# ======================================================================
# BLOCKS-STYLE STREAMLIT UI
# ======================================================================
# =========================================================
# LOCAL STORAGE + BLOCKS-STYLE STREAMLIT UI
# =========================================================
import json
from datetime import datetime
from typing import Dict, Any

# ---------- LOCAL STORAGE HELPERS ----------
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
    # dict: {ticker: notes_str}
    st.session_state.notes_store = _load_json(NOTES_FILE, {})
if "theses_store" not in st.session_state:
    # list of dicts
    st.session_state.theses_store = _load_json(THESES_FILE, [])


# ======================================================================
# ANALYSIS WRAPPER (hooks into your existing pipeline)
# ======================================================================
def run_equity_analysis(ticker: str, max_peers: int = 6) -> Dict[str, Any]:
    """
    Wraps analyze_ticker_pro + valuation + charts into a single object
    that the UI can reuse across pages.
    """
    ticker = ticker.upper().strip()

    # --- MODIFICATION: Wrap call in try/except ---
    try:
        (
            scored,
            text_synopsis_md,
            metrics_summary_md,
            focus_row,          # Changed (now a dict or None)
            news_md,
        ) = analyze_ticker_pro(ticker, peer_cap=max_peers)
    except Exception as e:
        logging.error(f"analyze_ticker_pro failed for {ticker}: {e}", exc_info=True)
        # Re-raise a more informative error to be caught by the UI
        raise Exception(f"Failed to analyze {ticker}. Ticker may be invalid or data unavailable. (Internal error: {e})")
    # --- END MODIFICATION ---


    # --- 1) High-level company summary markdown (split into two) ---
    overview_md = (
        "### Company Overview\n\n"
        + text_synopsis_md
        + "\n\n"
        + metrics_summary_md
    )
    
    # --- MODIFICATION: `ratings_md` is no longer created here ---
    # ratings_md = ( ... ) -> REMOVED

    # --- 2) Raw metrics row for the focus ticker ---
    raw_metrics = None
    if isinstance(scored, pd.DataFrame) and not scored.empty:
        if "Ticker" in scored.columns:
            focus = scored[scored["Ticker"] == ticker]
            if not focus.empty:
                raw_metrics = focus
            else:
                raw_metrics = scored.head(1)
        else:
            raw_metrics = scored.head(1)

    # --- 3) Valuation scenarios (Base/Bull/Bear) via your DCF/multiples engine ---
    valuation: Dict[str, Dict[str, Any]] = {}
    
    # Store the actual parameters from the base run
    base_metrics = get_metrics(ticker)
    base_params = _build_scenario_params(base_metrics, "Base")
    
    for scen in ["Base", "Bull", "Bear"]:
        try:
            df_val, md_val = _scenario_valuation_core(ticker, max_peers, scen)
        except Exception as e:
            df_val, md_val = pd.DataFrame(), f"_Error in {scen} scenario: {e}_"
        valuation[scen] = {"df": df_val, "md": md_val}


    # --- 4) Charts (5d price + EV/EBITDA vs growth scatter) ---
    fig_comp, fig_scatter, _ = charts(scored, ticker)
    fig_price = five_day_price_plot(ticker)

    return {
        "ticker": ticker,
        "overview_md": overview_md, # Added
        # "factor_percentiles": factor_percentiles, # REMOVED
        "focus_row": focus_row,                 # Kept (now a dict or None)
        "peers_df": scored,
        "valuation": valuation,
        "news_md": news_md,
        "raw_metrics": raw_metrics,
        "charts": {"price": fig_price, "scatter": fig_scatter},
        "base_params": base_params, # Add base parameters to results
        "current_price": get_price_and_shares(ticker)[0] # Get current price
    }


# ======================================================================
# GLOBAL STYLING (Blocks-style + nicer sidebar)
# ======================================================================
# ======================================================================
# GLOBAL STYLING (Blocks-style + nicer sidebar)
# ======================================================================
def inject_global_css():
    st.markdown(
        """
        <style>
        /* --- HIDE SIDEBAR --- */
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* --- GLOBAL TITLE --- */
        h1 {
            color: #0d1117 !important; /* Default to black/navy */
        }
        
        /* --- MAIN APP STYLING (LIGHT THEME DEFAULT) --- */
        .stApp {
            background: #ffffff;  
            color: #213547; /* Dark Navy text */
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        }
        header[data-testid="stHeader"] {background: transparent;}
        footer {visibility: hidden;}

        /* --- TOP NAVIGATION TABS (LIGHT THEME) --- */
        /* --- MODIFICATION: Full-width black bar --- */
        div[data-testid="stRadio"] {
            background: #0d1117; /* Black background */
            padding: 0;  
            
            /* --- NEW: Full-width "bust-out" --- */
            position: relative;
            width: 100vw; /* 100% of viewport width */
            left: 50%;
            right: 50%;
            margin-left: -50vw;
            margin-right: -50vw;
        }
        
        /* --- MODIFICATION: Centered, right-aligned buttons --- */
        div[data-testid="stRadio"] > div {
             /* This is the flex container for the buttons */
             gap: 8px;
             max-width: 1100px; /* Center the buttons with the content */
             margin: 0 auto; /* Center the button group */
             justify-content: flex-end; /* Align buttons to the right */
        }

        /* --- MODIFICATION: Light text for buttons --- */
        div[data-testid="stRadio"] label {
            display: inline-block;
            padding: 12px 16px;
            margin: 0;
            border-radius: 0;
            background: transparent;
            color: #8b949e; /* Inactive tab color (light grey) */
            border-bottom: 3px solid transparent;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        /* Hide the radio circle */
        div[data-testid="stRadio"] label > div:first-child {
            display: none;
        }
        
        div[data-testid="stRadio"] label span {
             font-size: 14px;
             font-weight: 500;
        }

        /* Hover style for inactive tabs */
        div[data-testid="stRadio"] label:hover {
            background: rgba(139, 148, 158, 0.1); /* Faint light hover */
            color: #f0f6fc; /* White text on hover */
        }

        /* Selected tab style */
        div[data-testid="stRadio"] label[data-checked="true"] {
            background: transparent;
            color: #f0f6fc; /* Active tab color (white) */
            border-bottom: 3px solid #3b82f6; /* Blue accent line */
        }
        /* --- End Top Nav --- */

        /* --- FACTOR BAR CHART STYLES (LIGHT THEME) --- */
        .factor-bar-container {
            margin-bottom: 12px;
        }
        .factor-bar-label {
            font-size: 14px;
            color: #213547; /* Navy text */
            margin-bottom: 6px;
        }
        .factor-bar-score {
            float: right;
            font-weight: 500;
            color: #0d1117; /* Black/navy text */
        }
        .factor-bar-bg {
            width: 100%;
            height: 10px;
            background-color: #e1e4e8; /* Light grey bar background */
            border-radius: 5px;
            overflow: hidden;
            position: relative; /* Added for centering line */
        }
        /* Center line for z-score */
        .factor-bar-bg::before {
            content: '';
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 2px;
            background-color: #ffffff; /* White background */
            z-index: 1; /* Above bg, below fill */
        }
        .factor-bar-fill-positive {
            height: 100%;
            background-color: #3fb950; /* Green for positive */
            border-radius: 0 5px 5px 0;
            transition: width 0.5s ease-in-out;
            position: absolute;
            left: 50%;
        }
        .factor-bar-fill-negative {
            height: 100%;
            background-color: #f85149; /* Red for negative */
            border-radius: 5px 0 0 5px;
            transition: width 0.5s ease-in-out;
            position: absolute;
            right: 50%; /* Anchor to the right of center */
        }

        .methodology-card {
            padding: 18px 20px;
            border-radius: 12px;
            background: #f9f9f9; /* Light card */
            border: 1px solid #e1e4e8; /* Light border */
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* Light shadow */
            height: 100%;
        }
        .methodology-card h4 {
            font-size: 16px;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 12px;
            color: #0d1117; /* Dark title */
        }
        .methodology-card p {
            font-size: 13px;
            color: #213547; /* MODIFIED: Dark Navy text */
            margin-bottom: 4px;
        }
        .methodology-card strong {
            color: #213547; /* Navy text */
            font-weight: 600;
        }
        /* --- End Factor Bar --- */

        /* --- CARD STYLING (LIGHT MODE) --- */
        .hero-card, .section-card, .kpi-card-new {
            border-radius: 12px; /* Sharper corners */
            padding: 28px 28px 24px 28px;
            background: #f9f9f9; /* Off-white card */
            border: 1px solid #e1e4e8; /* Light grey border */
            box-shadow: 0 8px 24px rgba(0,0,0,0.05); /* Lighter shadow */
            color: #213547; /* Navy text */
        }
        .kpi-card-new {
             padding: 18px 20px;
             margin-bottom: 16px;
        }
        .section-card {
            padding: 18px 20px;
            height: 100%;
        }
        
        .hero-title, .section-title, .kpi-value-new {
            font-size: 30px;  
            font-weight: 600;
            letter-spacing: 0.01em;
            margin-bottom: 4px;
            color: #0d1117; /* Black/darkest navy title */
        }
        .section-title {
            font-size: 16px;
        }
        .kpi-value-new {
             font-size: 28px;
        }
        
        .hero-subtitle, .section-subtitle, .kpi-label-new {
            font-size: 14px;
            color: #213547; /* MODIFIED: Dark Navy text */
            margin-bottom: 20px;
        }
        .section-subtitle {
             margin-bottom: 12px;
        }
        .kpi-label-new {
            margin-bottom: 4px;
        }

        /* Metric colors (Green/Red is standard for finance) */
        .positive-metric { color: #228B22; } /* Darker Green for light bg */
        .negative-metric { color: #D90429; } /* Darker Red for light bg */

        /* Button Styling - Blue Accent */
        .stButton > button {
            border-radius: 8px; /* Sharper */
            padding: 0.45rem 1.3rem;
            font-weight: 500;
            border: 1px solid #1c64f2; /* Blue border */
            background: linear-gradient(135deg, #3b82f6, #1d4ed8); /* Blue gradient */
            color: #ffffff; /* White text */
        }
        .stButton > button:hover {
            border-color: #60a5fa;
            filter: brightness(1.1);
            color: #ffffff;
        }
        
        /* Input boxes (Light Theme) */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
             border-radius: 8px !important;
             background: #ffffff !important;
             border: 1px solid #d1d5db !important;
             color: #0d1117 !important;
        }
        /* Placeholder text for light mode */
        .stTextInput input::placeholder {
            color: #4a5568; /* MODIFIED: Darker placeholder */
        }
        /* Command bar input on light */
        .stTextInput input[type="text"] {
            border-radius: 999px !important;
            padding-left: 14px;
            font-family: "SF Mono", ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            text-transform: uppercase;
        }
        
        /* Dataframe styling (Light Theme) */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e1e4e8;
        }

        /* Custom styles for Valuation Page (Light Theme) */
        .valuation-metric-label {
            font-size: 16px;
            color: #213547; /* MODIFIED: Dark Navy text */
            margin-bottom: 5px;
            font-weight: 400;
        }
        .valuation-metric-value {
            font-size: 38px; /* Larger for implied price */
            font-weight: 600;
            color: #0d1117;
            line-height: 1.2;
        }
        .valuation-current-price {
            font-size: 14px;
            color: #213547; /* MODIFIED: Dark Navy text */
            margin-top: 10px;
        }
        .valuation-upside {
            font-size: 14px;
            font-weight: 500;
        }
        .valuation-upside.positive {
            color: #228B22; /* Darker Green */
        }
        .valuation-upside.negative {
            color: #D90429; /* Darker Red */
        }
        .scenario-button-group .stRadio > label {
            border: 1px solid #d1d5db;
            background: #ffffff;
            border-radius: 8px;
            padding: 8px 12px;
            margin-right: 5px;
        }
        .scenario-button-group .stRadio > label:hover {
            background: #f9f9f9;
        }
        .scenario-button-group .stRadio > label[data-baseweb="radio"] > div:first-child {
            display: none;
        }
        .scenario-button-group .stRadio > label span {
            color: #213547;
        }
        .scenario-button-group .stRadio > label[data-checked="true"] {
            border-color: #d4af37;
            background: #fefce8; /* Light yellow */
        }
        
        /* Make sure number input arrows are styled */
        .stNumberInput div[data-testid="stVerticalBlock"] > div:last-child {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================
# PAGES
# ======================================================================

# ---------- DASHBOARD ----------
# ---------- DASHBOARD ----------
def render_dashboard():
    # --- MODIFICATION: Removed the light-theme override CSS block ---
    # It is no longer needed as this is now the default.

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Equity Research Platform</div>
            <div class="hero-subtitle">Analyze companies, build valuations, and create investment theses.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    
    col_search, col_btn = st.columns([4, 1])

    with col_search:
        ticker = st.text_input(
            "Search ticker symbol (e.g., AAPL, MSFT).",
            key="ticker_input",
            label_visibility="collapsed",
            placeholder="ENTER TICKER SYMBOL TO ANALYZE (E.G., AAPL)",
        ).upper()

    with col_btn:
        st.write("")
        analyze_clicked = st.button("Analyze", use_container_width=True)

    # max_peers = st.slider("Max peers to compare", 2, 15, 6, key="max_peers_dashboard") # <-- REMOVED
    max_peers = 6 # Set default since slider is removed

    if analyze_clicked and ticker:
        with st.spinner(f"Analyzing {ticker.upper()}..."):
            try:
                results = run_equity_analysis(ticker, max_peers=max_peers)
                st.session_state.last_results = results
                st.session_state.recent_tickers.insert(
                    0,
                    {"ticker": results["ticker"], "time": datetime.now().strftime("%Y-%m-%d %H:%M")},
                )
                st.session_state.recent_tickers = st.session_state.recent_tickers[:12]
                st.success(f"Analysis updated for {results['ticker']}")
                # Set the ticker input for valuation page
                st.session_state.valuation_ticker_input = results['ticker']
            except Exception as e:
                st.session_state.last_results = None # Clear last results
                logging.error(f"Error during analysis for {ticker}: {e}", exc_info=True)
                st.error(f"Analysis failed for {ticker.upper()}. The ticker might be invalid, delisted, or have no data. Error: {e}")


    st.write("")
    
    companies_tracked = len({x["ticker"] for x in st.session_state.recent_tickers}) or 0
    active_theses = len(st.session_state.theses_store)
    research_docs = len(st.session_state.notes_store)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card-new">
                <div class="kpi-label-new">Companies Tracked</div>
                <div class="kpi-value-new">{companies_tracked}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card-new">
                <div class="kpi-label-new">Active Theses</div>
                <div class="kpi-value-new">{active_theses}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card-new">
                <div class="kpi-label-new">Research Documents</div>
                <div class="kpi-value-new">{research_docs}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    st.write("")
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Quick Actions</div>
            <div class="section-subtitle">Common workflows</div>
        """,
        unsafe_allow_html=True,
    )

    qa_col1, qa_col2, qa_col3 = st.columns(3)

    with qa_col1:
        if st.button("Start New Analysis", use_container_width=True):
            st.session_state.top_nav_radio = "📈  Analysis"
            st.rerun() # Rerun to switch page
    with qa_col2:
        if st.button("Draft Thesis", use_container_width=True):
            st.session_state.top_nav_radio = "📝  Theses"
            st.rerun() # Rerun to switch page
    with qa_col3:
        if st.button("Open Research Notes", use_container_width=True):
            st.session_state.top_nav_radio = "📚  Research"
            st.rerun() # Rerun to switch page

    st.markdown("</div>", unsafe_allow_html=True)


    # --- MODIFICATION: "Recently Analyzed" section removed ---
    # st.write("")
    # st.markdown( ... ) -> This block was removed

    # --- MODIFICATION: "Recently Analyzed" section removed ---
    # st.write("")
    # st.markdown( ... ) -> This block was removed


# ---------- ANALYSIS PAGE ----------
def render_analysis_page():
    # inject_global_css() # Removed, now global in main()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Company Analysis</div>
            <div class="hero-subtitle">Review fundamental metrics, peer comparisons, and recent news.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    ticker = st.text_input(
        "Enter stock ticker (e.g., AAPL):",
        key="analysis_ticker",
        value=st.session_state.get("ticker_input", ""),
    ).upper()
    max_peers = st.slider("Max peers to compare", 2, 15, 6, key="max_peers_analysis")

    if st.button("Run Analysis", key="run_analysis_btn"):
        if ticker:
            with st.spinner(f"Analyzing {ticker.upper()}..."):
                # --- MODIFICATION: Wrap in try/except ---
                try:
                    results = run_equity_analysis(ticker, max_peers=max_peers)
                    st.session_state.last_results = results
                    st.session_state.recent_tickers.insert(
                        0,
                        {"ticker": results["ticker"], "time": datetime.now().strftime("%Y-%m-%d %H:%M")},
                    )
                    st.session_state.recent_tickers = st.session_state.recent_tickers[:12]
                    st.success(f"Analysis complete for {results['ticker']}.")
                    # Set the ticker input for valuation page
                    st.session_state.valuation_ticker_input = results['ticker']
                except Exception as e:
                    st.session_state.last_results = None # Clear last results
                    logging.error(f"Error during analysis for {ticker}: {e}", exc_info=True)
                    st.error(f"Analysis failed for {ticker.upper()}. The ticker might be invalid, delisted, or have no data. Error: {e}")
                # --- END MODIFICATION ---
        else:
            st.warning("Please enter a ticker symbol.")

    if st.session_state.last_results:
        res = st.session_state.last_results
        
        # --- NEW: Two-column layout for Overview and Ratings ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                """
                <div class="section-card">
                """,
                unsafe_allow_html=True
            )
            # overview_md already contains "### Company Overview"
            st.markdown(res.get("overview_md", "Overview not available."), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # --- MODIFICATION: Render new Factor Score Breakdown (Z-Score) ---
            st.markdown(
                f"""
                <div class="section-card">
                    <div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; color: #0d1117;">
                        Factor Score Breakdown
                    </div>
                    <div style="font-size: 14px; color: #4a5568; margin-top: -1rem; margin-bottom: 1.5rem;">
                        Individual factor analysis across key dimensions (z-score)
                    </div>
                """,
                unsafe_allow_html=True
            )
            
            focus_row = res.get("focus_row") # This is now a dict
            if focus_row is not None:
                for factor in FACTOR_BUCKETS.keys():
                    z_score = focus_row.get(factor) # .get() works on dicts
                    score_display = f"{z_score:+.2f}" if (z_score is not None and np.isfinite(z_score)) else "N/A"
                    
                    # Map z-score (-3 to +3) to a 0-100% range for the bar
                    # A z-score of 0 is 50%. A z-score of 3 is 100%. A z-score of -3 is 0%.
                    bar_width_pct = max(0, min(100, (z_score + 3) / 6 * 100)) if (z_score is not None and np.isfinite(z_score)) else 0
                    
                    fill_class = "factor-bar-fill-positive" if (z_score is not None and z_score > 0) else "factor-bar-fill-negative"
                    width_val = (abs(z_score) / 3) * 50 if (z_score is not None and np.isfinite(z_score)) else 0 # Scale to 50% max (3 std devs)
                    width_val = max(0, min(50, width_val))

                    st.markdown(
                        f"""
                        <div class="factor-bar-container">
                            <div class="factor-bar-label">
                                {factor}
                                <span class="factor-bar-score">{score_display}</span>
                            </div>
                            <div class="factor-bar-bg">
                                <div class="{fill_class}" style="width: {width_val}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("_Factor scores not available._")

            st.markdown("</div>", unsafe_allow_html=True)
            # --- END MODIFICATION ---

        # --- NEW: Factor Calculation Methodology Card ---
        st.write("") # Spacer
        st.markdown(
            """
            <div class="methodology-card">
                <h4>Factor Calculation Methodology</h4>
                <p>
                    Each factor score is an industry-neutral <strong>z-score</strong> based on a blend of underlying metrics. 
                    A score of 0 is average, +1 is one standard deviation above average, and -1 is one standard deviation below.
                </p>
                <p><strong>Valuation:</strong>
                    Blend of P/E Ratio, P/B Ratio, EV/EBITDA, and Price/Sales. 
                    Lower multiples result in a higher score.
                </p>
                <p><strong>Quality:</strong>
                    Blend of ROE (Return on Equity), Gross Margin, EBITDA Margin, and Interest Coverage. 
                    Higher profitability and coverage result in a higher score.
                </p>
                <p><strong>Growth:</strong>
                    Blend of TTM Revenue Growth (YoY) and Quarterly Asset Growth. 
                    Higher growth rates result in a higher score.
                </p>
                <p><strong>Momentum:</strong>
                    Blend of 12-Month Total Return and 5-Day Price vs. VWAP. 
                    Stronger recent price performance results in a higher score.
                </p>
                <p><strong>Leverage:</strong>
                    Based on Debt-to-Equity ratio. 
                    Lower leverage (less debt) results in a higher score.
                </p>
                <p><strong>Efficiency:</strong>
                    Blend of Gross Profitability (Gross Profit / Avg. Assets) and Accruals %. 
                    Higher profitability and lower accruals result in a higher score.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        # --- END NEW SECTION ---

        
        st.write("")
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Peer Analysis Table</div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(res["peers_df"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Key Metrics (Raw)</div>
            """,
            unsafe_allow_html=True
        )
        if isinstance(res["raw_metrics"], pd.DataFrame):
            st.dataframe(res["raw_metrics"], use_container_width=True)
        else:
            st.markdown("_Raw metrics view not available yet._")
        st.markdown("</div>", unsafe_allow_html=True)


        # --- MODIFICATION: Charts section removed ---
        # st.write("")
        # st.markdown( ... "Charts" ... ) -> This block was removed
        # --- END MODIFICATION ---


# ---------- VALUATION PAGE (NEW STREAMLIT VERSION) ----------
def render_valuation_page():
    # inject_global_css() # Removed, now global in main()
    import numpy as np
    import pandas as pd


    # Hero card
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

    # ---------- Small helpers (local to this page) ----------
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
        """
        Very simple DCF per-share:
        - Use EPS_TTM as proxy for FCFE per share.
        - Project for N years at growth_rate_pct.
        - Terminal value at year N with terminal_growth_pct.
        - Discount at discount_rate_pct.
        """
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

    # ---------- Select Company Card ----------
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Select Company</div>
            <div class="section-subtitle">Choose a company to value</div>
        """,
        unsafe_allow_html=True,
    )

    # Default ticker: last analyzed ticker if available
    default_ticker = ""
    if "valuation_ticker" in st.session_state:
        default_ticker = st.session_state.valuation_ticker
    elif st.session_state.get("last_results"):
        default_ticker = st.session_state.last_results.get("ticker", "")

    col_ticker, col_btn = st.columns([3, 1])
    with col_ticker:
        selected_ticker = st.text_input(
            "Ticker symbol",
            value=default_ticker,
            key="valuation_ticker_input",
            placeholder="Enter ticker symbol (e.g., AAPL, MSFT)",
        ).upper()
    with col_btn:
        load_clicked = st.button("Load Company", use_container_width=True)

    # Load company metrics when button clicked
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
            except Exception as e:
                logging.error(f"Failed to load metrics for {selected_ticker}: {e}")
                st.error(f"Could not load data for {selected_ticker.upper()}.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # Pull what we have in session
    metrics = st.session_state.get("valuation_metrics")
    current_ticker = st.session_state.get("valuation_ticker")
    current_price = st.session_state.get("valuation_price")

    if metrics and current_ticker:
        # ---------- Company Snapshot ----------
        prof = get_profile(current_ticker) or {}
        company_name = prof.get("name") or current_ticker
        eps = metrics.get("EPS_TTM")

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Company Snapshot</div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Ticker", current_ticker)
        with c2:
            st.metric("Company", company_name)
        with c3:
            st.metric("Current Price", format_currency(current_price or np.nan))

        if metrics.get("MarketCap"):
            mc = metrics["MarketCap"]
            if np.isfinite(mc):
                st.caption(f"Approx. market cap: ${mc:,.0f}M")

        if eps is None or not np.isfinite(eps):
            st.warning(
                "EPS_TTM is not available. DCF and P/E valuations may be limited or N/A."
            )

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        # ---------- Scenario Selection ----------
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Scenario Selection</div>
                <div class="section-subtitle">Choose valuation scenario assumptions</div>
            """,
            unsafe_allow_html=True,
        )

        scenario_presets = {
            "Bull Case": {"revenue_growth": 25.0, "discount_rate": 8.0, "terminal_growth": 4.0, "pe_multiple": 35.0},
            "Base Case": {"revenue_growth": 15.0, "discount_rate": 10.0, "terminal_growth": 3.0, "pe_multiple": 25.0},
            "Bear Case": {"revenue_growth": 5.0,  "discount_rate": 12.0, "terminal_growth": 2.0, "pe_multiple": 15.0},
        }

        selected_scenario = st.radio(
            "Scenario",
            ["Bull Case", "Base Case", "Bear Case"],
            index=1,
            horizontal=True,
            key="valuation_scenario_radio",
        )

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        # ---------- Valuation Assumptions ----------
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
            revenue_growth = st.number_input(
                "Revenue / EPS Growth (%)",
                value=float(preset["revenue_growth"]),
                step=1.0,
                format="%.1f",
                key="val_rev_growth",
            )
        with col_b:
            discount_rate = st.number_input(
                "Discount Rate / WACC (%)",
                value=float(preset["discount_rate"]),
                step=0.5,
                format="%.1f",
                key="val_discount_rate",
            )
        with col_c:
            terminal_growth = st.number_input(
                "Terminal Growth Rate (%)",
                value=float(preset["terminal_growth"]),
                step=0.5,
                format="%.1f",
                key="val_terminal_growth",
            )
        with col_d:
            pe_multiple = st.number_input(
                "Target P/E Multiple",
                value=float(preset["pe_multiple"]),
                step=1.0,
                format="%.1f",
                key="val_pe_multiple",
            )

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        # ---------- Run valuation ----------
        if st.button("Run Valuation", use_container_width=True, key="run_valuation_btn"):
            with st.spinner(f"Running valuation for {current_ticker} ({selected_scenario})..."):
                # DCF valuation
                dcf_price = compute_dcf_price(
                    eps=eps if eps is not None else np.nan,
                    growth_rate_pct=revenue_growth,
                    discount_rate_pct=discount_rate,
                    terminal_growth_pct=terminal_growth,
                    years=5,
                )

                # P/E valuation
                if eps is not None and np.isfinite(eps) and eps > 0:
                    pe_val = eps * pe_multiple
                else:
                    pe_val = np.nan

                # P/S valuation: if we have P/S (raw), back into price
                ps_val = np.nan
                ps_raw = metrics.get("P/S (raw)")
                if (
                    ps_raw is not None
                    and np.isfinite(ps_raw)
                    and ps_raw > 0
                    and metrics.get("MarketCap")
                    and np.isfinite(metrics["MarketCap"])
                ):
                    # Value per share = P/S * Sales_per_share
                    # MarketCap = P/S * Sales_TTM => implied price ≈ current_price * (target P/S / current P/S)
                    # Here we proxy target P/S from P/E multiple (loose, but keeps it simple)
                    if current_price and np.isfinite(current_price):
                        target_ps = ps_raw * (pe_multiple / preset["pe_multiple"])  # scale vs base case
                        ps_val = current_price * (target_ps / ps_raw)

                dcf_upside = compute_upside(dcf_price, current_price)
                pe_upside = compute_upside(pe_val, current_price)
                ps_upside = compute_upside(ps_val, current_price)

                # ---------- Valuation Results ----------
                st.markdown(
                    """
                    <div class="section-card">
                        <div class="section-title">Valuation Results</div>
                    """,
                    unsafe_allow_html=True,
                )

                # DCF
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

                st.markdown("<hr style='border-top: 1px solid #e1e4e8;'>", unsafe_allow_html=True)

                # P/E
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
                    st.markdown("<div class='valuation-metric-value'>N/A</div>", unsafe_allow_html=True)

                st.markdown("<hr style='border-top: 1px solid #e1e4e8;'>", unsafe_allow_html=True)

                # P/S-ish valuation
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
                        f"<span style='float:right'>{format_percent(ps_upside)}</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div class='valuation-metric-value'>N/A</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
                st.write("")

                # ---------- Sensitivity Analysis ----------
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
                    # Format as currency for display
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



# ---------- RESEARCH PAGE (PERSISTENT NOTES) ----------
def render_research_page():
    # inject_global_css() # Removed, now global in main()
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


# ---------- THESES PAGE (PERSISTENT THESES) ----------
def render_theses_page():
    # inject_global_css() # Removed, now global in main()
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
            <div classs="section-card">
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
def main():
    st.set_page_config(
        page_title="Equity Research Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",  # Collapse it, CSS will hide it
    )
    
    # --- NEW: Global Font Import ---
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:wght@400&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True
    )

    # --- TOP NAVIGATION BAR (MOVED) ---
    # This element is now at the very top, so its CSS can go full-width.
    page = st.radio(
        "Navigation",
        [
            "  |Dashboard|",
            "  |Analysis|",
            "  |Valuation|",
            "  |Research|",
            "  |Theses|",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="top_nav_radio",
    )
    
    # --- GLOBAL TITLE (MOVED) ---
    # This is now *inside* the centered div
    
    # Inject CSS *after* the radio button to ensure it can be styled
    # This function now contains the default LIGHT theme
    inject_global_css()

    # --- Center content + workspace badge ---
    st.markdown(
        "<div style='max-width:1100px;margin:0 auto;'>",
        unsafe_allow_html=True,
    )
    
    # --- GLOBAL TITLE (MOVED HERE) ---
    st.markdown(
        f"""
        <h1 style='text-align: center; margin-bottom: 1rem; padding-top: 1rem; 
                   font-weight: 400; font-family: "DM Serif Display", serif;
                   font-size: 2.75rem; 
                   color: #0d1117;'> 
            Equity Research Platform
        </h1>
        """, 
        unsafe_allow_html=True
    )


    # --- REMOVED: Workspace Badge ---
    # st.markdown( ... "Fricano Capital" ... )

    # Strip emoji prefix to route
    if "Dashboard" in page:
        render_dashboard()
    elif "Analysis" in page:
        render_analysis_page()
    elif "Valuation" in page:
        render_valuation_page()
    elif "Research" in page:
        render_research_page()
    elif "Theses" in page:
        render_theses_page()

    # Close centering div
    st.markdown("</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
