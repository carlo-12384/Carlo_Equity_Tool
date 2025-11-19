"""
Market-Learning Intuition Engine

This module combines:
1) A transparent, rule-based stock scoring engine (fundamentals, technicals,
   market context, intuition), and
2) A learning layer that trains on historical outcomes (realized returns)
   to adjust future scores based on what has actually worked.

USAGE (high level):

    from market_learning_engine import (
        FundamentalSnapshot,
        TechnicalSnapshot,
        MarketContext,
        IntuitionInputs,
        score_stock,           # deterministic
        MarketOutcomeLearner,  # learning layer
    )

    # 1) Use deterministic model normally
    fundamentals = FundamentalSnapshot(...)
    technicals = TechnicalSnapshot(...)
    context = MarketContext(...)
    intuition = IntuitionInputs(...)
    base_result = score_stock(fundamentals, technicals, context, intuition, ticker="AAPL")

    # 2) Log historical examples once you know outcomes
    learner = MarketOutcomeLearner()  # default history file: prediction_history.csv
    learner.add_training_example(
        ticker="AAPL",
        as_of_date="2025-01-01",
        horizon_days=90,
        fundamentals=fundamentals,
        technicals=technicals,
        market_context=context,
        intuition=intuition,
        realized_return=0.12,       # +12% over horizon
        benchmark_return=0.03,      # +3% for benchmark
    )

    # 3) Train or update the learning model
    learner.train_model(min_samples=40)

    # 4) For NEW stocks, get a score that includes the learned adjustment
    live_result = learner.predict_with_learning(
        fundamentals=fundamentals,
        technicals=technicals,
        market_context=context,
        intuition=intuition,
        ticker="AAPL",
    )

    print(live_result.final_score, live_result.rating)
    print(live_result.subscores.details["learning_layer"])
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import math
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# =====================================================================
# DATA CLASSES
# =====================================================================

@dataclass
class FundamentalSnapshot:
    revenue_cagr_3y: Optional[float] = None          # e.g. 0.15 = 15%
    gross_margin_trend_pp: Optional[float] = None    # percentage points over ~3y
    ebit_margin_trend_pp: Optional[float] = None
    fcf_margin: Optional[float] = None               # 0.12 = 12%
    roe: Optional[float] = None                      # 0.18 = 18%
    net_debt_to_ebitda: Optional[float] = None
    share_count_cagr: Optional[float] = None         # >0 dilution, <0 buybacks
    dso_trend_days: Optional[float] = None           # +ve = worse collections
    inventory_trend_days: Optional[float] = None
    insider_ownership_change_pct: Optional[float] = None  # +0.03 = +3%
    founder_led: bool = False
    accounting_red_flags: int = 0                    # count of known issues


@dataclass
class TechnicalSnapshot:
    rel_strength_3m: Optional[float] = None          # vs benchmark, 1.10 = +10%
    rel_strength_12m: Optional[float] = None
    volatility_3m: Optional[float] = None            # annualized, e.g. 0.25
    avg_daily_dollar_volume: Optional[float] = None  # liquidity
    drawdown_from_high_pct: Optional[float] = None   # -0.20 = 20% below high
    price_above_200d_ma_pct: Optional[float] = None  # 0.10 = 10% above 200d MA


@dataclass
class MarketContext:
    market_regime: str = "NEUTRAL"   # "RISK_ON", "RISK_OFF", "VOLATILE", "NEUTRAL"
    sector_tailwind_score: float = 0.5  # 0–1, higher = stronger tailwinds
    macro_risk_score: float = 0.5       # 0–1, higher = more macro risk


@dataclass
class IntuitionInputs:
    # 0–10 sliders (human intuition layer)
    analyst_conviction: int = 5
    narrative_clarity: int = 5
    management_quality: int = 5
    business_understandability: int = 5

    # Qualitative flags
    red_flags: List[str] = None   # e.g. ["customer_concentration"]
    green_flags: List[str] = None # e.g. ["network_effects"]

    # Manual override: -5..+5, used in rule-based layer
    analyst_override: float = 0.0

    def __post_init__(self):
        if self.red_flags is None:
            self.red_flags = []
        if self.green_flags is None:
            self.green_flags = []


@dataclass
class ScoreBreakdown:
    fundamentals: float
    technicals: float
    market_context: float
    intuition: float
    details: Dict[str, Any]


@dataclass
class FinalScoreResult:
    ticker: Optional[str]
    final_score: float          # -100..100
    rating: str                 # e.g. "Strong Buy"
    subscores: ScoreBreakdown


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# =====================================================================
# FUNDAMENTAL SCORING (RULE-BASED)
# =====================================================================

def score_fundamentals(f: FundamentalSnapshot) -> (float, Dict[str, float]):
    details: Dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    def add_signal(name: str, score: float, weight: float):
        nonlocal total_weight, weighted_sum
        details[name] = score
        weighted_sum += score * weight
        total_weight += weight

    # Revenue growth (3y CAGR)
    if f.revenue_cagr_3y is not None:
        g = f.revenue_cagr_3y
        if g >= 0.25:
            s = 30
        elif g >= 0.15:
            s = 20
        elif g >= 0.05:
            s = 10
        elif g >= 0.0:
            s = 0
        elif g >= -0.05:
            s = -10
        else:
            s = -20
        add_signal("revenue_growth", s, weight=1.5)

    # Gross margin trend
    if f.gross_margin_trend_pp is not None:
        t = f.gross_margin_trend_pp
        if t >= 5:
            s = 20
        elif t >= 2:
            s = 10
        elif t >= 0:
            s = 0
        elif t >= -2:
            s = -10
        else:
            s = -20
        add_signal("gross_margin_trend", s, weight=1.2)

    # EBIT margin trend
    if f.ebit_margin_trend_pp is not None:
        t = f.ebit_margin_trend_pp
        if t >= 4:
            s = 15
        elif t >= 1:
            s = 8
        elif t >= 0:
            s = 0
        elif t >= -2:
            s = -10
        else:
            s = -20
        add_signal("ebit_margin_trend", s, weight=1.2)

    # FCF margin
    if f.fcf_margin is not None:
        m = f.fcf_margin
        if m >= 0.20:
            s = 25
        elif m >= 0.10:
            s = 15
        elif m >= 0.05:
            s = 5
        elif m >= 0.0:
            s = 0
        elif m >= -0.05:
            s = -10
        else:
            s = -20
        add_signal("fcf_margin", s, weight=1.3)

    # ROE
    if f.roe is not None:
        r = f.roe
        if r >= 0.25:
            s = 20
        elif r >= 0.15:
            s = 12
        elif r >= 0.10:
            s = 5
        elif r >= 0.0:
            s = 0
        else:
            s = -10
        add_signal("roe", s, weight=1.0)

    # Net debt / EBITDA
    if f.net_debt_to_ebitda is not None:
        d = f.net_debt_to_ebitda
        if d <= 0:
            s = 15
        elif d <= 1:
            s = 10
        elif d <= 2:
            s = 5
        elif d <= 3:
            s = 0
        elif d <= 4:
            s = -10
        else:
            s = -20
        add_signal("leverage", s, weight=1.1)

    # Share count CAGR
    if f.share_count_cagr is not None:
        sc = f.share_count_cagr
        if sc <= -0.03:
            s = 15
        elif sc <= -0.01:
            s = 8
        elif sc <= 0.0:
            s = 3
        elif sc <= 0.03:
            s = -5
        else:
            s = -15
        add_signal("share_count_cagr", s, weight=1.0)

    # DSO trend
    if f.dso_trend_days is not None:
        dso = f.dso_trend_days
        if dso <= -5:
            s = 10
        elif dso <= -1:
            s = 5
        elif dso <= 1:
            s = 0
        elif dso <= 5:
            s = -10
        else:
            s = -20
        add_signal("dso_trend", s, weight=0.7)

    # Inventory trend
    if f.inventory_trend_days is not None:
        inv = f.inventory_trend_days
        if inv <= -5:
            s = 8
        elif inv <= -1:
            s = 4
        elif inv <= 1:
            s = 0
        elif inv <= 5:
            s = -8
        else:
            s = -15
        add_signal("inventory_trend", s, weight=0.6)

    # Insider ownership change
    if f.insider_ownership_change_pct is not None:
        ch = f.insider_ownership_change_pct
        if ch >= 0.05:
            s = 15
        elif ch >= 0.01:
            s = 8
        elif ch >= 0.0:
            s = 0
        elif ch >= -0.02:
            s = -8
        else:
            s = -15
        add_signal("insider_ownership_change", s, weight=1.0)

    # Founder-led
    if f.founder_led:
        add_signal("founder_led", 8, weight=0.7)

    # Accounting red flags
    if f.accounting_red_flags > 0:
        s = -10 * f.accounting_red_flags
        s = _clamp(s, -30, 0)
        add_signal("accounting_red_flags", s, weight=1.2)

    if total_weight == 0:
        return 0.0, details

    composite = weighted_sum / total_weight
    composite = _clamp(composite, -100, 100)
    return composite, details


# =====================================================================
# TECHNICAL SCORING (RULE-BASED)
# =====================================================================

def score_technicals(t: TechnicalSnapshot) -> (float, Dict[str, float]):
    details: Dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    def add_signal(name: str, score: float, weight: float):
        nonlocal total_weight, weighted_sum
        details[name] = score
        weighted_sum += score * weight
        total_weight += weight

    # Relative strength 3m
    if t.rel_strength_3m is not None:
        rs = t.rel_strength_3m - 1.0
        if rs >= 0.15:
            s = 20
        elif rs >= 0.05:
            s = 10
        elif rs >= 0.0:
            s = 0
        elif rs >= -0.05:
            s = -8
        else:
            s = -18
        add_signal("rel_strength_3m", s, weight=1.2)

    # Relative strength 12m
    if t.rel_strength_12m is not None:
        rs = t.rel_strength_12m - 1.0
        if rs >= 0.30:
            s = 25
        elif rs >= 0.10:
            s = 12
        elif rs >= 0.0:
            s = 0
        elif rs >= -0.10:
            s = -10
        else:
            s = -20
        add_signal("rel_strength_12m", s, weight=1.3)

    # Volatility
    if t.volatility_3m is not None:
        vol = t.volatility_3m
        if vol <= 0.18:
            s = 10
        elif vol <= 0.25:
            s = 5
        elif vol <= 0.35:
            s = 0
        elif vol <= 0.50:
            s = -10
        else:
            s = -20
        add_signal("volatility_3m", s, weight=0.8)

    # Liquidity
    if t.avg_daily_dollar_volume is not None:
        liq = t.avg_daily_dollar_volume
        if liq >= 20_000_000:
            s = 15
        elif liq >= 5_000_000:
            s = 8
        elif liq >= 1_000_000:
            s = 3
        elif liq >= 250_000:
            s = 0
        else:
            s = -10
        add_signal("liquidity", s, weight=1.0)

    # Drawdown from high
    if t.drawdown_from_high_pct is not None:
        dd = t.drawdown_from_high_pct
        if dd >= -0.10:
            s = 5
        elif dd >= -0.25:
            s = 0
        elif dd >= -0.40:
            s = -5
        else:
            s = -15
        add_signal("drawdown_from_high", s, weight=0.9)

    # Price vs 200d MA
    if t.price_above_200d_ma_pct is not None:
        p = t.price_above_200d_ma_pct
        if p >= 0.15:
            s = 15
        elif p >= 0.05:
            s = 8
        elif p >= 0.0:
            s = 0
        elif p >= -0.10:
            s = -8
        else:
            s = -15
        add_signal("price_vs_200d_ma", s, weight=1.0)

    if total_weight == 0:
        return 0.0, details

    composite = weighted_sum / total_weight
    composite = _clamp(composite, -100, 100)
    return composite, details


# =====================================================================
# MARKET CONTEXT SCORING (RULE-BASED)
# =====================================================================

def score_market_context(ctx: MarketContext) -> (float, Dict[str, float]):
    details: Dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    def add_signal(name: str, score: float, weight: float):
        nonlocal total_weight, weighted_sum
        details[name] = score
        weighted_sum += score * weight
        total_weight += weight

    regime = ctx.market_regime.upper()
    if regime == "RISK_ON":
        regime_score = 15
    elif regime == "RISK_OFF":
        regime_score = -15
    elif regime == "VOLATILE":
        regime_score = -5
    else:
        regime_score = 0
    add_signal("market_regime", regime_score, weight=1.0)

    st = _clamp(ctx.sector_tailwind_score, 0.0, 1.0)
    tailwind_score = st * 20 - 5
    add_signal("sector_tailwind", tailwind_score, weight=1.2)

    mr = _clamp(ctx.macro_risk_score, 0.0, 1.0)
    macro_score = (1 - mr) * 20 - 5
    add_signal("macro_risk", macro_score, weight=1.2)

    if total_weight == 0:
        return 0.0, details

    composite = weighted_sum / total_weight
    composite = _clamp(composite, -100, 100)
    return composite, details


# =====================================================================
# INTUITION SCORING (RULE-BASED)
# =====================================================================

def score_intuition(inp: IntuitionInputs) -> (float, Dict[str, float]):
    details: Dict[str, float] = {}

    def slider_to_score(x: int, center: float = 5.0, spread: float = 5.0, scale: float = 20.0) -> float:
        delta = (x - center) / spread
        return _clamp(delta * scale, -scale, scale)

    conviction_score = slider_to_score(inp.analyst_conviction)
    narrative_score = slider_to_score(inp.narrative_clarity)
    mgmt_score = slider_to_score(inp.management_quality)
    understanding_score = slider_to_score(inp.business_understandability)

    details["analyst_conviction"] = conviction_score
    details["narrative_clarity"] = narrative_score
    details["management_quality"] = mgmt_score
    details["business_understandability"] = understanding_score

    green_score = len(inp.green_flags) * 3.0
    red_score = -len(inp.red_flags) * 4.0

    details["green_flags"] = green_score
    details["red_flags"] = red_score

    base = (
        conviction_score * 0.35
        + narrative_score * 0.25
        + mgmt_score * 0.25
        + understanding_score * 0.15
        + green_score * 0.5
        + red_score * 0.6
    )
    base = _clamp(base, -60, 60)

    override = _clamp(inp.analyst_override, -5.0, 5.0) * 4.0
    details["analyst_override_effect"] = override

    final = _clamp(base + override, -100, 100)
    return final, details


# =====================================================================
# FINAL AGGREGATION (RULE-BASED BASELINE)
# =====================================================================

def _rating_from_score(score: float) -> str:
    if score >= 60:
        return "Strong Buy"
    elif score >= 35:
        return "Buy"
    elif score >= 15:
        return "Positive Bias / Accumulate"
    elif score >= -15:
        return "Hold / Neutral"
    elif score >= -35:
        return "Reduce / Cautious"
    else:
        return "Avoid / Strong Sell"


def score_stock(
    fundamentals: FundamentalSnapshot,
    technicals: TechnicalSnapshot,
    market_context: MarketContext,
    intuition: IntuitionInputs,
    ticker: Optional[str] = None,
) -> FinalScoreResult:
    f_score, f_details = score_fundamentals(fundamentals)
    t_score, t_details = score_technicals(technicals)
    m_score, m_details = score_market_context(market_context)
    i_score, i_details = score_intuition(intuition)

    subscores = ScoreBreakdown(
        fundamentals=f_score,
        technicals=t_score,
        market_context=m_score,
        intuition=i_score,
        details={
            "fundamentals": f_details,
            "technicals": t_details,
            "market_context": m_details,
            "intuition": i_details,
        },
    )

    w_f = 0.45
    w_t = 0.20
    w_m = 0.15
    w_i = 0.20

    final_score = (
        f_score * w_f +
        t_score * w_t +
        m_score * w_m +
        i_score * w_i
    )
    final_score = _clamp(final_score, -100, 100)
    rating = _rating_from_score(final_score)

    return FinalScoreResult(
        ticker=ticker,
        final_score=final_score,
        rating=rating,
        subscores=subscores,
    )


# =====================================================================
# MARKET OUTCOME LEARNING LAYER
# =====================================================================

class MarketOutcomeLearner:
    """
    Learns from realized market outcomes.

    - Stores training examples in a CSV history file
    - Trains a RandomForestRegressor to map features -> future alpha
    - Adjusts the base rule-based score using learned expected alpha

    Target = realized_return - benchmark_return  (i.e., alpha)
    """

    def __init__(self, history_path: str = "prediction_history.csv"):
        self.history_path = history_path
        self.model: Optional[RandomForestRegressor] = None
        self.feature_columns: Optional[List[str]] = None

        if os.path.exists(self.history_path):
            try:
                df = pd.read_csv(self.history_path)
                self.feature_columns = [
                    c for c in df.columns
                    if c not in {"ticker", "as_of_date", "horizon_days",
                                 "realized_return", "benchmark_return", "target_alpha"}
                ]
            except Exception:
                # If file is corrupted or unreadable, ignore
                self.feature_columns = None

    # ------- Feature builder -------

    def _encode_regime(self, ctx: MarketContext) -> float:
        mapping = {
            "RISK_ON": 1.0,
            "RISK_OFF": -1.0,
            "VOLATILE": 0.5,
            "NEUTRAL": 0.0,
        }
        return mapping.get(ctx.market_regime.upper(), 0.0)

    def _build_features(
        self,
        fundamentals: FundamentalSnapshot,
        technicals: TechnicalSnapshot,
        market_context: MarketContext,
        intuition: IntuitionInputs,
    ) -> Dict[str, float]:
        """
        Turn input snapshots into a flat numeric feature dict.
        Includes both raw metrics and rule-based subscores.
        """
        # First compute rule-based scores (they are good features)
        f_score, _ = score_fundamentals(fundamentals)
        t_score, _ = score_technicals(technicals)
        m_score, _ = score_market_context(market_context)
        i_score, _ = score_intuition(intuition)

        features: Dict[str, float] = {
            # Rule-based composite features
            "fundamental_score": f_score,
            "technical_score": t_score,
            "market_score": m_score,
            "intuition_score": i_score,

            # Fundamental raw-ish features
            "rev_cagr_3y": fundamentals.revenue_cagr_3y or 0.0,
            "gm_trend_pp": fundamentals.gross_margin_trend_pp or 0.0,
            "ebit_trend_pp": fundamentals.ebit_margin_trend_pp or 0.0,
            "fcf_margin": fundamentals.fcf_margin or 0.0,
            "roe": fundamentals.roe or 0.0,
            "net_debt_ebitda": fundamentals.net_debt_to_ebitda or 0.0,
            "share_count_cagr": fundamentals.share_count_cagr or 0.0,
            "dso_trend_days": fundamentals.dso_trend_days or 0.0,
            "inventory_trend_days": fundamentals.inventory_trend_days or 0.0,
            "insider_own_change": fundamentals.insider_ownership_change_pct or 0.0,
            "founder_led_flag": 1.0 if fundamentals.founder_led else 0.0,
            "accounting_red_flags": float(fundamentals.accounting_red_flags),

            # Technical features
            "rs_3m": (technicals.rel_strength_3m or 1.0) - 1.0,
            "rs_12m": (technicals.rel_strength_12m or 1.0) - 1.0,
            "vol_3m": technicals.volatility_3m or 0.0,
            "avg_dollar_vol": technicals.avg_daily_dollar_volume or 0.0,
            "dd_from_high": technicals.drawdown_from_high_pct or 0.0,
            "px_vs_200d": technicals.price_above_200d_ma_pct or 0.0,

            # Market context features
            "regime_code": self._encode_regime(market_context),
            "sector_tailwind": _clamp(market_context.sector_tailwind_score, 0.0, 1.0),
            "macro_risk": _clamp(market_context.macro_risk_score, 0.0, 1.0),

            # Intuition raw sliders
            "conviction": float(intuition.analyst_conviction),
            "narrative_clarity": float(intuition.narrative_clarity),
            "management_quality": float(intuition.management_quality),
            "business_understandability": float(intuition.business_understandability),
            "num_green_flags": float(len(intuition.green_flags)),
            "num_red_flags": float(len(intuition.red_flags)),
            "override": float(intuition.analyst_override),
        }

        return features

    # ------- Training data logging -------

    def add_training_example(
        self,
        ticker: str,
        as_of_date: str,
        horizon_days: int,
        fundamentals: FundamentalSnapshot,
        technicals: TechnicalSnapshot,
        market_context: MarketContext,
        intuition: IntuitionInputs,
        realized_return: float,
        benchmark_return: Optional[float] = None,
    ) -> None:
        """
        Log a single training example into the history CSV.

        realized_return: actual stock return over the horizon (e.g., 0.12 = +12%)
        benchmark_return: corresponding benchmark return; if None, target = realized_return
        """
        features = self._build_features(fundamentals, technicals, market_context, intuition)

        if benchmark_return is None:
            target_alpha = realized_return
        else:
            target_alpha = realized_return - benchmark_return

        row = {
            "ticker": ticker,
            "as_of_date": as_of_date,
            "horizon_days": int(horizon_days),
            "realized_return": float(realized_return),
            "benchmark_return": float(benchmark_return) if benchmark_return is not None else 0.0,
            "target_alpha": float(target_alpha),
        }
        row.update(features)

        new_df = pd.DataFrame([row])

        if os.path.exists(self.history_path):
            try:
                existing = pd.read_csv(self.history_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
            except Exception:
                combined = new_df
        else:
            combined = new_df

        combined.to_csv(self.history_path, index=False)
        self.feature_columns = [c for c in combined.columns
                                if c not in {"ticker", "as_of_date", "horizon_days",
                                             "realized_return", "benchmark_return", "target_alpha"}]

    # ------- Model training -------

    def train_model(self, min_samples: int = 40) -> bool:
        """
        Train (or retrain) the RandomForest model on all logged history.

        Returns True if training succeeded and the model is ready to use,
        otherwise False (e.g., not enough samples).
        """
        if not os.path.exists(self.history_path):
            return False

        df = pd.read_csv(self.history_path)
        if "target_alpha" not in df.columns:
            return False

        df = df.dropna(subset=["target_alpha"])
        if len(df) < min_samples:
            return False

        feature_cols = [
            c for c in df.columns
            if c not in {"ticker", "as_of_date", "horizon_days",
                         "realized_return", "benchmark_return", "target_alpha"}
        ]
        if not feature_cols:
            return False

        X = df[feature_cols].values
        y = df["target_alpha"].values

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        self.model = model
        self.feature_columns = feature_cols
        return True

    # ------- Prediction with learning -------

    def predict_with_learning(
        self,
        fundamentals: FundamentalSnapshot,
        technicals: TechnicalSnapshot,
        market_context: MarketContext,
        intuition: IntuitionInputs,
        ticker: Optional[str] = None,
    ) -> FinalScoreResult:
        """
        Returns a FinalScoreResult where:
          - base rule-based score is computed
          - if a trained model exists, its expected alpha prediction is converted
            into a score adjustment and added to the base score.

        If no trained model or not enough data, this just returns the base result.
        """
        base_result = score_stock(fundamentals, technicals, market_context, intuition, ticker=ticker)

        # If we don't have a trained model, just return base
        if self.model is None or not self.feature_columns:
            # Add a note in details so you know it's pure rule-based
            base_result.subscores.details["learning_layer"] = {
                "active": False,
                "reason": "No trained model; using rule-based score only.",
            }
            return base_result

        features = self._build_features(fundamentals, technicals, market_context, intuition)

        # Align feature vector to training columns, fill missing with 0
        x_vec = np.array([features.get(col, 0.0) for col in self.feature_columns]).reshape(1, -1)

        try:
            pred_alpha = float(self.model.predict(x_vec)[0])
        except Exception:
            # If anything goes wrong, fall back to base
            base_result.subscores.details["learning_layer"] = {
                "active": False,
                "reason": "Prediction error; using rule-based score only.",
            }
            return base_result

        # Convert predicted alpha into a score adjustment.
        # Example: +10% alpha => about +30 score, capped at +/-40.
        adjustment = _clamp(pred_alpha * 300.0, -40.0, 40.0)

        learned_score = _clamp(base_result.final_score + adjustment, -100.0, 100.0)
        learned_rating = _rating_from_score(learned_score)

        # Extend subscores.details
        learning_info = {
            "active": True,
            "predicted_alpha": pred_alpha,
            "score_adjustment": adjustment,
            "base_final_score": base_result.final_score,
            "learned_final_score": learned_score,
        }

        base_result.subscores.details["learning_layer"] = learning_info

        # Return a new FinalScoreResult with the adjusted score & rating
        return FinalScoreResult(
            ticker=base_result.ticker,
            final_score=learned_score,
            rating=learned_rating,
            subscores=base_result.subscores,
        )


# =====================================================================
# SIMPLE SELF-TEST
# =====================================================================

if __name__ == "__main__":
    # Quick smoke test with dummy data
    fundamentals = FundamentalSnapshot(
        revenue_cagr_3y=0.18,
        gross_margin_trend_pp=3.0,
        ebit_margin_trend_pp=1.5,
        fcf_margin=0.14,
        roe=0.18,
        net_debt_to_ebitda=1.2,
        share_count_cagr=-0.015,
        dso_trend_days=0.5,
        inventory_trend_days=-1.0,
        insider_ownership_change_pct=0.02,
        founder_led=True,
        accounting_red_flags=0,
    )
    technicals = TechnicalSnapshot(
        rel_strength_3m=1.10,
        rel_strength_12m=1.25,
        volatility_3m=0.24,
        avg_daily_dollar_volume=8_000_000,
        drawdown_from_high_pct=-0.18,
        price_above_200d_ma_pct=0.06,
    )
    context = MarketContext(
        market_regime="RISK_ON",
        sector_tailwind_score=0.8,
        macro_risk_score=0.3,
    )
    intuition = IntuitionInputs(
        analyst_conviction=8,
        narrative_clarity=9,
        management_quality=8,
        business_understandability=9,
        red_flags=["customer_concentration"],
        green_flags=["network_effects", "recurring_revenue"],
        analyst_override=-1.0,
    )

    # Base score
    base = score_stock(fundamentals, technicals, context, intuition, ticker="TEST")
    print("Base Final Score:", round(base.final_score, 2), "| Rating:", base.rating)

    learner = MarketOutcomeLearner(history_path="prediction_history_demo.csv")

    # Add a few fake training examples
    learner.add_training_example(
        ticker="TEST",
        as_of_date="2025-01-01",
        horizon_days=90,
        fundamentals=fundamentals,
        technicals=technicals,
        market_context=context,
        intuition=intuition,
        realized_return=0.15,
        benchmark_return=0.05,
    )

    # Try training (will usually need more samples in real life)
    trained = learner.train_model(min_samples=1)
    print("Trained:", trained)

    live = learner.predict_with_learning(
        fundamentals, technicals, context, intuition, ticker="TEST"
    )
    print("Learned Final Score:", round(live.final_score, 2), "| Rating:", live.rating)
    print("Learning Layer Info:", live.subscores.details.get("learning_layer"))
