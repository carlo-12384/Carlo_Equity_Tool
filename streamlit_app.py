# -*- coding: utf-8 -*-
"""Carlo Equity Tool — Streamlit App"""

import os, time, math, logging, textwrap, datetime as dt
import requests, pandas as pd, numpy as np, yfinance as yf
import matplotlib.pyplot as plt
from functools import lru_cache
from typing import List
import streamlit as st

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
    mc  = row.get("MarketCap") if isinstance(row, (dict, pd.Series)) else None
    exch= prof.get("exchange") or prof.get("ticker") or ""
    mc_txt = "N/A"
    if isinstance(mc, (int, float)) and np.isfinite(mc) and mc > 0:
        # assume marketCap in millions; display as B where possible
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
        if not np.isfinite(value): return ""
        color_class = "positive-metric" if value >= 0 else "negative-metric"
        if invert_color: color_class = "negative-metric" if value >= 0 else "positive-metric"
        format_str = f"{{:.1f}}{{}}" if not is_percent else f"{{:+.1f}}{{}}"
        return f"<span class=\"{color_class}\">" + format_str.format(value, suffix) + "</span>"

    if np.isfinite(pe_raw): bits.append(f"- P/E: {format_metric(pe_raw, suffix='x', invert_color=True)}")
    if np.isfinite(pb_raw): bits.append(f"- P/B: {format_metric(pb_raw, suffix='x', invert_color=True)}")
    if np.isfinite(ps_raw): bits.append(f"- P/S: {format_metric(ps_raw, suffix='x', invert_color=True)}")
    if np.isfinite(ev_ebitda): bits.append(f"- EV/EBITDA: {format_metric(ev_ebitda, suffix='x', invert_color=True)}")
    if np.isfinite(gm): bits.append(f"- Gross Margin: {format_metric(gm, suffix='%', is_percent=True)}")
    if np.isfinite(em): bits.append(f"- EBITDA Margin: {format_metric(em, suffix='%', is_percent=True)}")
    if np.isfinite(rg): bits.append(f"- Revenue Growth (YoY, TTM): {format_metric(rg, suffix='%', is_percent=True)}")
    if np.isfinite(roe): bits.append(f"- ROE: {format_metric(roe, suffix='%', is_percent=True)}")
    if np.isfinite(dte): bits.append(f"- Debt/Equity: {format_metric(dte, suffix='x', invert_color=True)}")

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
        _ = net_prev_ttm  # currently unused, kept for future expansion

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
            st = last_q_from_rows(q_bs, ["Short Long Term Debt","Short Long Term Debt Total","Short/Current Long Term Debt"])
            total_debt = (lt if np.isfinite(lt) else 0) + (st if np.isfinite(st) else 0)
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

        # Normalized metrics (kept for completeness)
        pe  = (price / eps_ttm) if all(np.isfinite([price, eps_ttm])) and eps_ttm != 0 else np.nan
        pb  = (mktcap / total_equity) if all(np.isfinite([mktcap, total_equity])) and total_equity != 0 else np.nan
        ps  = (mktcap / rev_ttm) if all(np.isfinite([mktcap, rev_ttm])) and rev_ttm != 0 else np.nan
        ev_ebitda = (ev / ebitda_ttm) if all(np.isfinite([ev, ebitda_ttm])) and ebitda_ttm != 0 else np.nan
        _ = (pe, pb, ps, ev_ebitda)  # not used directly

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
        df["VWAP"]        = df["Ticker"].map(vwaps)
        df["Ret12m"]      = df["Ticker"].map(rets)

    scored = prepare_factors(df)

    focus_row = None
    if (not scored.empty) and ("Ticker" in scored.columns):
        m = scored["Ticker"] == ticker
        if m.any():
            focus_row = scored.loc[m].iloc[0]

    parts = build_waterfall_dict(focus_row) if focus_row is not None else {}
    peers_used = [p for p in peers if p in list(scored.get("Ticker", []))] if not scored.empty else []

    # Separate synopsis and KPI details
    text_synopsis_md = get_company_text_summary(ticker, focus_row if focus_row is not None else {})
    metrics_summary_md = get_company_metrics_summary(focus_row if focus_row is not None else {})

    kpi_details_lines = []
    if focus_row is not None:
        comp = focus_row.get("CompositeScore")
        if comp is not None and np.isfinite(comp):
            color_class = "positive-metric" if comp >= 0 else "negative-metric"
            kpi_details_lines.append(f"- **Composite Score:** <span class=\"{color_class}\">`{comp:.2f}`</span> (industry-neutral, winsorized z-blend)")
        else:
            kpi_details_lines.append("- **Composite Score:** `N/A`")
        for b in BUCKET_WEIGHTS:
            bv = focus_row.get(b)
            if bv is not None and np.isfinite(bv):
                color_class = "positive-metric" if bv >= 0 else "negative-metric"
                kpi_details_lines.append(f"  - {b}: <span class=\"{color_class}\">{bv:+.2f}</span> (weight {BUCKET_WEIGHTS[b]:.0%})")
        rf = focus_row.get("RiskFlags", "")
        if isinstance(rf, str) and rf.strip():
            kpi_details_lines.append(f"- **Risk Flags:** {rf.strip()}")
        kpi_details_lines.append(f"- **Peers:** {', '.join(peers_used) if peers_used else 'None'}")

    kpi_ratings_md = f"### Ratings for **{ticker}**\n" + ("\n".join(kpi_details_lines) if kpi_details_lines else "")

    waterfall_md = ""
    if parts:
        wf_lines = []
        for k, v in parts.items():
            if np.isfinite(v):
                color_class = "positive-metric" if v >= 0 else "negative-metric"
                wf_lines.append(f"- {k}: <span class=\"{color_class}\">{v:+.2f}</span> (weight {BUCKET_WEIGHTS[k]:.0%})")
            else:
                wf_lines.append(f"- {k}: `N/A` (weight {BUCKET_WEIGHTS[k]:.0%})")
        waterfall_md = "\n**Composite Waterfall (contribution):**\n" + "\n".join(wf_lines)

    news_items = get_company_news(ticker, n=8)
    news_md = "### Recent Headlines\n" + render_news_md(news_items)

    return scored, text_synopsis_md, metrics_summary_md, kpi_ratings_md, waterfall_md, news_md

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
                                  c=dfx["CompositeScore"], alpha=0.75)
                ax2.set_title("EV/EBITDA vs Revenue Growth (size≈MktCap, color=Composite)")
                ax2.set_xlabel("EV/EBITDA"); ax2.set_ylabel("Revenue Growth %")
                if focus in set(dfx["Ticker"]):
                    row = dfx[dfx["Ticker"] == focus].iloc[0]
                    ax2.annotate(focus, (row["EV/EBITDA"], row["RevenueGrowth%"]),
                                 xytext=(5, 5), textcoords="offset points")
                cb = fig2.colorbar(sc_); cb.set_label("Composite Score")
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

# -------------------- THESIS BUILDER HELPERS (BACKEND-ONLY) --------------------
def search_document(text: str, query: str) -> str:
    if not text or not query:
        return "_No query or document loaded._"
    lines = []
    q = query.lower().strip()
    for i, line in enumerate(text.splitlines(), 1):
        if q in line.lower():
            snippet = textwrap.shorten(line.strip(), width=160, placeholder="…")
            lines.append(f"- Line {i}: {snippet}")
    return "\n".join(lines) if lines else "_No matches found in this document._"

def save_document_to_library(lib: dict, title: str, text: str):
    """
    Pure-Python version: updates an in-memory dict {title: text} and returns
    (updated_library, list_of_titles).
    """
    if lib is None or not isinstance(lib, dict):
        lib = {}
    title = (title or "").strip()
    if title and text:
        lib = {**lib, title: text}
    choices = sorted(list(lib.keys()))
    return lib, choices

def load_document_from_library(lib: dict, title: str):
    if not lib or title not in lib:
        return "", ""
    return title, lib.get(title, "")

def search_library(lib: dict, query: str) -> str:
    if not lib or not query:
        return "_No library documents or query provided._"
    q = query.lower().strip()
    hits = []
    for name, text in lib.items():
        count = text.lower().count(q)
        if count > 0:
            snippet = textwrap.shorten(text.replace("\n", " "), width=220, placeholder="…")
            hits.append(f"- **{name}** — {count} hit(s)  \n  {snippet}")
    return "\n".join(hits) if hits else "_No matches found across the library._"

def update_notes_preview(notes: str) -> str:
    return notes or ""

def snapshot_chart_into_notes(notes: str, ticker: str):
    notes = notes or ""
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"\n\n> Snapshot **{ticker.upper()}** — Composite & peer visuals as of `{stamp}`\n"
    new_notes = notes + line
    return new_notes, new_notes

def metrics_snippet_for_ticker(ticker: str) -> str:
    m = get_metrics(ticker.upper())
    if not m:
        return ""
    gm = m.get("GrossMargin%")
    rg = m.get("RevenueGrowth%")
    pe = m.get("P/E (raw)")
    ev_eb = m.get("EV/EBITDA")
    dte = m.get("DebtToEquity")
    bits = [f"- **Ticker:** `{ticker.upper()}`"]
    if np.isfinite(pe): bits.append(f"- P/E: **{pe:.1f}x**")
    if np.isfinite(ev_eb): bits.append(f"- EV/EBITDA: **{ev_eb:.1f}x**")
    if np.isfinite(gm): bits.append(f"- Gross Margin: **{gm:.1f}%**")
    if np.isfinite(rg): bits.append(f"- Revenue Growth (YoY): **{rg:+.1f}%**")
    if np.isfinite(dte): bits.append(f"- Debt/Equity: **{dte:.2f}x**")
    return "\n".join(bits) + "\n"

def insert_metrics_into_notes(notes: str, ticker: str):
    base = notes or ""
    snippet = metrics_snippet_for_ticker(ticker)
    if not snippet:
        return base, base
    new_notes = base + ("\n\n" if base else "") + snippet
    return new_notes, new_notes

def load_net_income_series(symbol: str) -> pd.Series:
    q_is, _, _ = yf_quarterly(symbol)
    s = _first_row(q_is, [
        "Net Income Common Stockholders","Net Income Applicable To Common Shares","Net Income","NetIncome"
    ])
    return _coerce_cols_desc(s)

def load_eps_series(symbol: str) -> pd.Series:
    ni = load_net_income_series(symbol)
    price, shares = get_price_and_shares(symbol)
    if not np.isfinite(shares) or shares <= 0 or ni.empty:
        return pd.Series([], dtype=float)
    return ni / shares

def term_lookup_metrics(ticker: str, term: str):
    term = (term or "").strip()
    ticker = (ticker or "").upper().strip()
    if not ticker or not term:
        return "_No ticker or term provided._", None

    latest_val = np.nan
    series = pd.Series([], dtype=float)
    label = term

    if term == "Revenue":
        series = load_revenue_series(ticker)
        label = "Revenue"
    elif term == "Net Income":
        series = load_net_income_series(ticker)
        label = "Net Income"
    elif term == "EPS (TTM)":
        m = get_metrics(ticker)
        latest_val = m.get("EPS_TTM", np.nan)
        label = "EPS (TTM)"
    elif term == "Gross Margin":
        m = get_metrics(ticker)
        gm = m.get("GrossMargin%", np.nan)
        latest_val = gm if np.isfinite(gm) else np.nan
        label = "Gross Margin %"
    elif term == "EBITDA":
        m = get_metrics(ticker)
        gm = m.get("EBITDAMargin%", np.nan)
        latest_val = gm if np.isfinite(gm) else np.nan
        series = load_revenue_series(ticker)
        label = "EBITDA (proxy trend via revenue)"

    if series is not None and not series.empty:
        latest_val = float(series.iloc[0])

    if not np.isfinite(latest_val):
        md = f"_No reliable time series found for **{label}** on `{ticker}`._"
        return md, None

    md = f"**{label}** for `{ticker}`  \nLatest value: **{latest_val:,.2f}** (approx; free data)"

    fig = None
    if series is not None and not series.empty:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        s = series.sort_index()
        ax.plot(s.index, s.values)
        ax.set_title(f"{ticker} — {label} trend")
        ax.set_xlabel("Period")
        ax.set_ylabel(label)

    return md, fig

# -------------------- COMPARISON BAR HELPERS --------------------
def render_pins_md(pins: List[str]) -> str:
    if not pins:
        return "_No companies pinned yet._"
    return "Pinned companies: " + ", ".join(f"`{p}`" for p in pins)

def build_pins_plot(pins: List[str]):
    if not pins:
        return None
    labels, vals = [], []
    for p in pins:
        m = get_metrics(p)
        gm = m.get("GrossMargin%")
        if gm is not None and np.isfinite(gm):
            labels.append(p)
            vals.append(gm)
    if not labels:
        return None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(labels, vals)
    ax.set_ylabel("Gross Margin %")
    ax.set_title("Pinned Companies — Gross Margin %")
    return fig

def add_pin(pins: List[str], ticker: str):
    if pins is None or not isinstance(pins, list):
        pins = []
    t = (ticker or "").upper().strip()
    if t:
        if t not in pins:
            pins = pins + [t]
        if len(pins) > 4:
            pins = pins[-4:]
    return pins, render_pins_md(pins), build_pins_plot(pins)

def clear_pins():
    pins: List[str] = []
    return pins, render_pins_md(pins), None

# -------------------- VALUATION / SCENARIO HELPERS --------------------
def _estimate_fcff_and_net_debt(symbol: str):
    """
    Very rough FCFF and net-debt estimate from free yfinance statements.
    Returns (fcff_ttm, net_debt).
    """
    symbol = symbol.upper().strip()
    try:
        q_is, q_bs, q_cf = yf_quarterly(symbol)

        # EBIT & depreciation
        ebit_ttm = ttm_from_rows(q_is, ["Ebit","EBIT","Operating Income","OperatingIncome"])
        dep_ttm  = ttm_from_rows(q_cf, ["Depreciation","Depreciation And Amortization","Depreciation Amortization Depletion"])

        # Capex (usually negative in CF; take abs)
        capex_raw = ttm_from_rows(q_cf, [
            "Capital Expenditure","CapitalExpenditure","Capital Expenditures",
            "Purchase Of Property Plant Equipment","PurchaseOfPPE"
        ])
        capex_ttm = abs(capex_raw) if np.isfinite(capex_raw) else (0.6 * dep_ttm if np.isfinite(dep_ttm) else np.nan)

        # Tax rate
        tax_exp_ttm = ttm_from_rows(q_is, ["Income Tax Expense","Income Tax Provision","Provision For Income Taxes"])
        pretax_ttm  = ttm_from_rows(q_is, [
            "Pretax Income","PretaxIncome",
            "Income Before Tax","Earnings From Continuing Operations Before Income Taxes"
        ])
        if all(np.isfinite([tax_exp_ttm, pretax_ttm])) and pretax_ttm != 0:
            tr = tax_exp_ttm / pretax_ttm
            tax_rate = float(np.clip(tr, 0.0, 0.35))
        else:
            tax_rate = 0.21  # generic US-ish

        if not np.isfinite(ebit_ttm):
            # Fallback: use Net Income as proxy if EBIT missing
            netinc_ttm = ttm_from_rows(q_is, [
                "Net Income Common Stockholders",
                "Net Income Applicable To Common Shares",
                "Net Income","NetIncome"
            ])
            ebit_ttm = netinc_ttm if np.isfinite(netinc_ttm) else np.nan

        if not np.isfinite(ebit_ttm):
            return np.nan, np.nan

        fcff_ttm = (ebit_ttm * (1.0 - tax_rate)) + (dep_ttm if np.isfinite(dep_ttm) else 0.0) - (capex_ttm if np.isfinite(capex_ttm) else 0.0)

        # Net debt
        total_debt = last_q_from_rows(q_bs, ["Total Debt","TotalDebt"])
        if not np.isfinite(total_debt):
            lt = last_q_from_rows(q_bs, ["Long Term Debt","LongTermDebt","Long Term Debt Noncurrent"])
            st = last_q_from_rows(q_bs, ["Short Long Term Debt","Short Long Term Debt Total","Short/Current Long Term Debt"])
            total_debt = (lt if np.isfinite(lt) else 0) + (st if np.isfinite(st) else 0)
            if total_debt == 0:
                total_debt = np.nan

        cash = last_q_from_rows(q_bs, [
            "Cash And Cash Equivalents","Cash",
            "Cash And Cash Equivalents And Short Term Investments"
        ])

        net_debt = np.nan
        if np.isfinite(total_debt) and np.isfinite(cash):
            net_debt = total_debt - cash

        # sanity check: if FCFF is absurdly negative relative to revenue, drop it
        rev_series = load_revenue_series(symbol)
        if np.isfinite(fcff_ttm) and rev_series is not None and not rev_series.empty:
            rev_ttm = float(rev_series.iloc[:4].sum())
            if rev_ttm > 0 and fcff_ttm < -0.5 * rev_ttm:
                fcff_ttm = np.nan

        return float(fcff_ttm), float(net_debt) if np.isfinite(net_debt) else np.nan
    except Exception as e:
        logging.warning(f"_estimate_fcff_and_net_debt failed for {symbol}: {e}")
        return np.nan, np.nan


def _build_scenario_params(metrics, scenario):
    rev_g = metrics.get("RevenueGrowth%", np.nan)
    base_g = rev_g/100 if np.isfinite(rev_g) else 0.05

    if scenario == "Bull":
        return dict(
            name="Bull",
            growth = min(base_g + 0.10, 0.25),   # big bump (10%)
            discount = 0.07,                    # 7% discount rate
            terminal_growth = 0.03,             # 3% long term
            multiple_adj = 1.75                 # 75% multiple expansion
        )
    elif scenario == "Bear":
        return dict(
            name="Bear",
            growth = max(base_g - 0.05, 0.00),
            discount = 0.14,                    # 14% WACC
            terminal_growth = 0.01,
            multiple_adj = 0.60                 # 40% compression
        )
    else: # Base
        return dict(
            name="Base",
            growth = base_g,
            discount = 0.09,
            terminal_growth = 0.02,
            multiple_adj = 1.00
        )


def _scenario_valuation_core(ticker: str, max_peers: int, scenario: str):
    """
    Main engine: builds DCF + P/E + P/S price targets for a scenario.
    Returns (DataFrame, explanation_md).
    """
    ticker = ticker.upper().strip()
    if not ticker:
        return pd.DataFrame(), "_No ticker provided._"

    m = get_metrics(ticker)
    if not m:
        return pd.DataFrame(), f"_Could not load metrics for `{ticker}`._"

    price_now, shares = get_price_and_shares(ticker)
    if not (np.isfinite(price_now) and np.isfinite(shares) and shares > 0):
        return pd.DataFrame(), f"_Could not determine current price / share count for `{ticker}`._"

    fcff0, net_debt = _estimate_fcff_and_net_debt(ticker)
    scen = _build_scenario_params(m, scenario)

    rows = []
    notes = []

    # ---------- DCF (5y FCFF) ----------
    dcf_price = np.nan
    if np.isfinite(fcff0):

        # NEW RULE: If FCFF is negative, skip DCF completely
        if fcff0 < 0:
            notes.append(
                "- **DCF skipped:** FCFF is negative, making DCF not meaningful for early-stage or unprofitable companies."
            )
        else:
            g = scen["growth"]
            r = scen["discount"]
            g_term = scen["terminal_growth"]

            # Project FCFF 5 years
            fcffs = [(fcff0 * ((1.0 + g) ** t)) for t in range(1, 6)]

            # PV of FCFF
            pv_fcf = sum(f / ((1.0 + r) ** (i + 1)) for i, f in enumerate(fcffs))

            # Terminal Value
            tv = fcffs[-1] * (1.0 + g_term) / (r - g_term)
            pv_tv = tv / ((1.0 + r) ** 5)

            ev = pv_fcf + pv_tv
            equity_value = ev - net_debt if np.isfinite(net_debt) else ev
            dcf_price = equity_value / shares

            rows.append({
                "Method": f"DCF (5y FCFF)",
                "Scenario": scen["name"],
                "Implied Price": float(dcf_price),
                "Upside vs Current %": ((dcf_price / price_now) - 1.0) * 100 if np.isfinite(dcf_price) and price_now > 0 else np.nan
            })

    # ---------- PE multiple valuation ----------
    pe_price = np.nan
    pe_raw = m.get("P/E (raw)", np.nan)
    eps_ttm = m.get("EPS_TTM", np.nan)
    if np.isfinite(pe_raw) and np.isfinite(eps_ttm) and eps_ttm > 0 and price_now > 0:
        pe_target_multiple = pe_raw * scen["multiple_adj"]
        pe_price = pe_target_multiple * eps_ttm
        rows.append({
            "Method": f"P/E Multiple (target {pe_target_multiple:.1f}x)",
            "Scenario": scen["name"],
            "Implied Price": float(pe_price),
            "Upside vs Current %": ((pe_price / price_now) - 1.0) * 100
        })
    elif np.isfinite(eps_ttm) and eps_ttm <= 0:
        notes.append(f"- **P/E skipped:** {ticker} has negative or zero TTM EPS. P/E valuation is not applicable.")

    # ---------- PS multiple valuation ----------
    ps_price = np.nan
    ps_raw = m.get("P/S (raw)", np.nan)
    rev_ttm = ttm_from_rows(yf.Ticker(ticker).quarterly_income_stmt, [
        "Total Revenue","TotalRevenue","Revenue" ]
    )
    if np.isfinite(ps_raw) and np.isfinite(rev_ttm) and rev_ttm > 0 and shares > 0 and price_now > 0:
        ps_target_multiple = ps_raw * scen["multiple_adj"]
        ps_price = (ps_target_multiple * rev_ttm) / shares
        rows.append({
            "Method": f"P/S Multiple (target {ps_target_multiple:.1f}x)",
            "Scenario": scen["name"],
            "Implied Price": float(ps_price),
            "Upside vs Current %": ((ps_price / price_now) - 1.0) * 100
        })
    elif np.isfinite(rev_ttm) and rev_ttm <= 0:
        notes.append(f"- **P/S skipped:** {ticker} has negative or zero TTM Revenue. P/S valuation is not applicable.")

    df_out = pd.DataFrame(rows)
    explanation_md = f"### Valuation for {ticker} ({scen['name']} Scenario)\n\n" \
                     f"Current Price: **${price_now:.2f}**\n" \
                     + "\n".join(notes)

    return df_out, explanation_md

# -------------------- FRONTEND (STREAMLIT APP) --------------------

# Helper to convert dict to DataFrame for display
def dict_to_df(data: dict) -> pd.DataFrame:
    if not data: return pd.DataFrame()
    return pd.DataFrame([data])

def main_streamlit_app():
    st.set_page_config(layout="wide")
    st.title("Carlo Equity Tool")

    st.markdown("""
    This tool provides a rapid equity analysis combining fundamental metrics, peer comparison, and basic valuation scenarios.
    Data is sourced from Yahoo Finance and Finnhub (free tier). Performance may vary with data availability.
    """)

    # Initialize session state for notes and pins if they don't exist
    if "notes_text" not in st.session_state:
        st.session_state["notes_text"] = ""
    if "library" not in st.session_state:
        st.session_state["library"] = {}
    if "library_titles" not in st.session_state:
        st.session_state["library_titles"] = []
    if "pinned_tickers" not in st.session_state:
        st.session_state["pinned_tickers"] = []
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Company Summary"
    if "analysis_data" not in st.session_state:
        st.session_state["analysis_data"] = {}

    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page_selection = st.radio(
            "Go to",
            (
                "Company Summary",
                "Peer Analysis Table",
                "Valuation Scenarios",
                "Recent News",
                "Charts",
                "Research Library"
            ),
            key="page_radio"
        )
        st.session_state["current_page"] = page_selection

    # Input for ticker and peers (always visible)
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL)", "GOOG", key="ticker_input_main")
    max_peers_input = st.slider("Max Peers to Compare", min_value=1, max_value=10, value=6, key="max_peers_input_main")
    run_analysis_button = st.button("Run Analysis", key="run_analysis_button_main")

    if run_analysis_button and ticker_input:
        with st.spinner(f"Analyzing {ticker_input.upper()} and its peers..."):
            scored_df, text_synopsis, metrics_summary, kpi_ratings, waterfall, news = analyze_ticker_pro(
                ticker_input, max_peers=max_peers_input
            )
            # Compute view_df and figures once and store
            cols = [
                "Ticker","Industry","MarketCap","Latest Price","VWAP","Ret12m",
                "P/E (raw)","P/B (raw)","EV/EBITDA (raw)","P/S (raw)",
                "ROE%","GrossMargin%","EBITDAMargin%","RevenueGrowth%","DebtToEquity",
                "GrossProfitability","AssetGrowth%","Accruals%","InterestCoverage",
                "Valuation","Quality","Growth","Leverage","Momentum","Efficiency",
                "CompositeScore","RiskFlags"
            ]
            view_df = (
                scored_df[[c for c in cols if c in scored_df.columns]]
                .sort_values("CompositeScore", ascending=False, na_position="last")
                if (not scored_df.empty) and ("CompositeScore" in scored_df.columns)
                else pd.DataFrame(columns=cols)
            )

            # Format numerical columns for display (for view_df)
            dollar_cols = ["MarketCap", "Latest Price", "VWAP"]
            percent_cols = ["ROE%", "GrossMargin%", "EBITDAMargin%", "RevenueGrowth%", "AssetGrowth%", "Accruals%", "Ret12m", "Mom_VWAP_Diff%"]
            two_decimal_cols = ["P/E (raw)", "P/B (raw)", "EV/EBITDA (raw)", "P/S (raw)",
                                "Valuation", "Quality", "Growth", "Leverage", "Momentum", "Efficiency",
                                "CompositeScore", "InterestCoverage", "GrossProfitability", "DebtToEquity"]

            for col in view_df.columns:
                if pd.api.types.is_numeric_dtype(view_df[col]):
                    if col in dollar_cols:
                        view_df[col] = view_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else np.nan)
                    elif col in percent_cols:
                        view_df[col] = view_df[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else np.nan)
                    elif col in two_decimal_cols:
                        view_df[col] = view_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else np.nan)

            fig1, fig2, fig3 = charts(scored_df if not scored_df.empty else pd.DataFrame(columns=["Ticker","CompositeScore"]),
                                     ticker_input.upper())

            st.session_state["analysis_data"] = {
                "ticker": ticker_input.upper(),
                "scored_df": scored_df,
                "view_df": view_df,
                "text_synopsis": text_synopsis,
                "metrics_summary": metrics_summary,
                "kpi_ratings": kpi_ratings,
                "waterfall": waterfall,
                "news": news,
                "fig1": fig1,
                "fig2": fig2,
                "fig3": fig3,
                "max_peers_input": max_peers_input # Store max_peers_input for valuation scenarios
            }
            st.session_state["valuation_results_Base"] = {}
            st.session_state["valuation_results_Bull"] = {}
            st.session_state["valuation_results_Bear"] = {}


    # Conditional rendering based on page selection
    current_analysis_data = st.session_state.get("analysis_data", {})
    ticker_analyzed = current_analysis_data.get("ticker", "")

    if st.session_state["current_page"] == "Company Summary":
        st.header(f"Company Summary for {ticker_analyzed}")
        if ticker_analyzed:
            st.markdown(current_analysis_data["text_synopsis"])
            st.markdown(current_analysis_data["metrics_summary"])
            st.markdown(current_analysis_data["kpi_ratings"])
            st.markdown(current_analysis_data["waterfall"])
            if st.button(f"Pin {ticker_analyzed} for Comparison", key="pin_summary"):
                st.session_state["pinned_tickers"], _, _ = add_pin(st.session_state["pinned_tickers"], ticker_analyzed)
                st.success(f"Pinned {ticker_analyzed}!")
        else:
            st.info("Run an analysis using the ticker input above.")

    elif st.session_state["current_page"] == "Peer Analysis Table":
        st.header(f"Peer Analysis Table for {ticker_analyzed} and its peers")
        if ticker_analyzed and not current_analysis_data["view_df"].empty:
            st.dataframe(current_analysis_data["view_df"].set_index("Ticker"))
        else:
            st.info("Run an analysis using the ticker input above to see peer comparison table.")

    elif st.session_state["current_page"] == "Valuation Scenarios":
        st.header(f"Valuation Scenarios for {ticker_analyzed}")
        if ticker_analyzed:
            max_peers_for_val = current_analysis_data.get("max_peers_input", 6)
            col_base, col_bull, col_bear = st.columns(3)
            scenarios = {"Base": col_base, "Bull": col_bull, "Bear": col_bear}
            for scen_name, col in scenarios.items():
                with col:
                    if st.button(f"Run {scen_name} Case Valuation", key=f"val_button_{scen_name}"):
                        with st.spinner(f"Running {scen_name} valuation for {ticker_analyzed}..."):
                            val_df, val_md = _scenario_valuation_core(ticker_analyzed, max_peers_for_val, scen_name)
                            st.session_state[f"valuation_results_{scen_name}"] = {"df": val_df, "md": val_md}
            
            for scen_name in scenarios.keys():
                if f"valuation_results_{scen_name}" in st.session_state and st.session_state[f"valuation_results_{scen_name}"]:
                    st.markdown(st.session_state[f"valuation_results_{scen_name}"]["md"])
                    if not st.session_state[f"valuation_results_{scen_name}"]["df"].empty:
                        st.dataframe(st.session_state[f"valuation_results_{scen_name}"]["df"].set_index("Method"))
        else:
            st.info("Run an analysis using the ticker input above to run valuation scenarios.")

    elif st.session_state["current_page"] == "Recent News":
        st.header(f"Recent News for {ticker_analyzed}")
        if ticker_analyzed:
            st.markdown(current_analysis_data["news"])
        else:
            st.info("Run an analysis using the ticker input above to see recent news.")

    elif st.session_state["current_page"] == "Charts":
        st.header(f"Charts for {ticker_analyzed}")
        if ticker_analyzed:
            if current_analysis_data.get("fig1") is not None: st.pyplot(current_analysis_data["fig1"])
            if current_analysis_data.get("fig2") is not None: st.pyplot(current_analysis_data["fig2"])
            if current_analysis_data.get("fig3") is not None: st.pyplot(current_analysis_data["fig3"])
        else:
            st.info("Run an analysis using the ticker input above to see charts.")

    elif st.session_state["current_page"] == "Research Library":
        st.header("Notes & Research Library")

        st.subheader("Pinned Companies (Gross Margin %)")
        st.markdown(render_pins_md(st.session_state["pinned_tickers"])) # Render pinned list
        if st.session_state["pinned_tickers"]:
            pinned_plot = build_pins_plot(st.session_state["pinned_tickers"])
            if pinned_plot: st.pyplot(pinned_plot)
            if st.button("Clear Pinned Companies", key="clear_pins_btn"): # Clear pins button
                st.session_state["pinned_tickers"], _, _ = clear_pins()
                st.experimental_rerun()

        st.subheader("Your Research Notes")
        st.session_state["notes_text"] = st.text_area("Enter your notes here:", st.session_state["notes_text"], height=300)

        st.write("### Notes Actions")
        note_action = st.radio("Choose an action:", ("None", "Snapshot Chart", "Insert Metrics"), key="note_action_radio")
        if note_action == "Snapshot Chart" and ticker_analyzed:
            if st.button("Add Current Charts Snapshot to Notes"):
                st.session_state["notes_text"], _ = snapshot_chart_into_notes(st.session_state["notes_text"], ticker_analyzed)
                st.success("Charts snapshot added to notes!")
        elif note_action == "Insert Metrics" and ticker_analyzed:
            if st.button("Add Current Metrics Snippet to Notes"):
                st.session_state["notes_text"], _ = insert_metrics_into_notes(st.session_state["notes_text"], ticker_analyzed)
                st.success("Metrics snippet added to notes!")

        st.write("### Research Library")
        new_doc_title = st.text_input("New Document Title:", key="new_doc_title")
        new_doc_content = st.text_area("New Document Content:", height=150, key="new_doc_content")
        if st.button("Save to Library", key="save_to_library_btn"): # Save button
            if new_doc_title and new_doc_content:
                st.session_state["library"], st.session_state["library_titles"] = save_document_to_library(st.session_state["library"], new_doc_title, new_doc_content)
                st.success(f"'{new_doc_title}' saved to library.")
            else:
                st.warning("Please provide both a title and content for the document.")

        if st.session_state["library_titles"]:
            selected_doc_title = st.selectbox("Load Document from Library:", ["-- Select --"] + st.session_state["library_titles"], key="load_doc_select")
            if selected_doc_title != "-- Select --":
                _, content = load_document_from_library(st.session_state["library"], selected_doc_title)
                st.markdown(f"**{selected_doc_title}**\n```\n{content}\n```")

        library_search_query = st.text_input("Search Library:", key="library_search_query")
        if library_search_query:
            search_results = search_library(st.session_state["library"], library_search_query)
            st.markdown("**Search Results:**")
            st.markdown(search_results)

# Run the Streamlit app if this script is executed directly
if __name__ == "__main__":
    # THIS IS FOR RUNNING IN COLAB. To run a Streamlit app in Colab, you typically need to use `!streamlit run <filename>`. However, the prompt has a `main_streamlit_app()` function which suggests it should be executable directly in a notebook cell.
    # Given the previous context and the error, it's likely the user intends for the Streamlit app to be run via this direct execution block, which is then captured by the colab runtime.
    try:
        main_streamlit_app()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logging.error(f"Streamlit app failed: {e}")
