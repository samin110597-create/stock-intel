"""Causal features, lightweight ensemble, and inspectable risk rules.

Scores measure agreement, never calibrated probabilities of making money.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mi.indicators import indicator_pack
from .data import ticker, validated


def features(prices: pd.DataFrame) -> pd.DataFrame:
    ind = indicator_pack(prices)
    out = pd.DataFrame(index=prices.index)
    for n in (1, 5, 20, 60):
        out[f'return_{n}'] = prices.close.pct_change(n)
    out['rsi'] = ind.rsi14 / 100
    out['macd'] = ind.macd_hist / ind.atr14
    out['trend'] = ind.dist_sma200
    out['ema_gap'] = ind.ema21 / ind.ema50 - 1
    out['atr_pct'] = ind.atr14 / prices.close
    out['volume'] = np.log1p(ind.rvol20)
    return out.replace([np.inf, -np.inf], np.nan)


def technical(ind: pd.Series, close: float) -> float:
    return float(np.mean([
        np.sign(close - ind.sma200), np.sign(ind.ema21 - ind.ema50),
        np.sign(ind.macd_hist), 1 if ind.rsi14 > 55 else -1 if ind.rsi14 < 45 else 0,
    ]))


def linear_score(prices: pd.DataFrame, horizon: int = 5) -> float:
    x = features(prices)
    # Last included label ends strictly before the observation date. The
    # future open/close cannot be used by this model before they exist.
    forward = prices.close.shift(-horizon) / prices.open.shift(-1) - 1
    usable = x.notna().all(axis=1) & forward.notna()
    usable.iloc[-(horizon + 1):] = False
    if usable.sum() < 250:
        raise ValueError("Linear model needs 250 mature training labels after warm-up.")
    y = (forward[usable] > 0).astype(int)
    if y.nunique() != 2:
        raise ValueError("Training labels contain only one direction.")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500, random_state=0))
    model.fit(x.loc[usable], y)
    return float(2 * model.predict_proba(x.iloc[[-1]])[0, 1] - 1)


def action(score: float) -> str:
    if not np.isfinite(score) or not -1 <= score <= 1:
        raise ValueError("Model score must be finite and between -1 and 1.")
    return 'LONG' if score >= .35 else 'SHORT' if score <= -.35 else 'WAIT'


def levels(price: float, atr: float, side: str) -> dict:
    if side == 'WAIT':
        return {'entry': None, 'stop': None, 'targets': []}
    sign = 1 if side == 'LONG' else -1
    risk = max(1.5 * atr, .01 * price)
    if not np.isfinite(risk) or risk <= 0 or price - 3 * risk <= 0:
        raise ValueError("Volatility is too large to construct sensible positive price levels.")
    return {'entry': round(price, 4), 'stop': round(price - sign * risk, 4),
            'targets': [round(price + sign * risk * r, 4) for r in (1, 2, 3)]}


def state_for_laya(prices: pd.DataFrame, ind: pd.Series, scores: dict) -> dict:
    # No ticker/date: reduce the opportunity to recall a historical outcome.
    # All facts are restricted to this observation's prefix.
    return {
        'task': 'Experimental five-session stock setup; future prices are unknown.',
        'close': round(float(prices.close.iloc[-1]), 4),
        'rsi14': round(float(ind.rsi14), 2),
        'above_sma200': bool(prices.close.iloc[-1] > ind.sma200),
        'ema21_above_ema50': bool(ind.ema21 > ind.ema50),
        'macd_histogram': round(float(ind.macd_hist), 4),
        'relative_volume': round(float(ind.rvol20), 3),
        'atr_percent': round(float(ind.atr14 / prices.close.iloc[-1] * 100), 3),
        'reversal_structure': bool(ind.choch),
        'close_above_prior_20_high': bool(prices.close.iloc[-1] > prices.high.iloc[-21:-1].max()),
        'close_below_prior_20_low': bool(prices.close.iloc[-1] < prices.low.iloc[-21:-1].min()),
        'scores_minus1_bearish_plus1_bullish': scores,
    }


def components(prices: pd.DataFrame, models=('linear',), horizon=5, full_decisions=True):
    if set(models) - {'linear', 'chronos', 'laya'}:
        raise ValueError('Supported models: linear, chronos, laya')
    ind = indicator_pack(prices).iloc[-1]
    scores = {'technical': technical(ind, prices.close.iloc[-1])}
    detail, unavailable = {}, {}
    for name in models:
        try:
            if name == 'linear':
                scores[name] = linear_score(prices, horizon)
            elif name == 'chronos':
                from .models import chronos_score
                scores[name], detail[name] = chronos_score(prices, horizon)
            elif name == 'laya':
                from .models import laya_decisions
                # Keep Laya input identical across ablations; the full model
                # is an ensemble of independent votes, not different prompts.
                scores[name], detail[name] = laya_decisions(state_for_laya(
                    prices, ind, {'technical': scores['technical']}), full=full_decisions)
            else:
                raise ValueError(f"Unknown model: {name}")
            if not np.isfinite(scores[name]) or abs(scores[name]) > 1:
                raise ValueError("Invalid model score")
        except Exception as exc:
            scores.pop(name, None)
            from mi.redact import clean
            unavailable[name] = str(clean(f'{type(exc).__name__}: {exc}'))
    return scores, detail, unavailable


@dataclass
class Decision:
    ticker: str
    current_price: float
    as_of: str
    direction: str
    action: str
    setup_quality: str
    reversal: str
    breakout: str
    entry: float | None
    stop: float | None
    targets: list[float]
    agreement_score: float
    components: dict
    model_details: dict
    unavailable: dict
    warnings: list[str]
    horizon_sessions: int = 5

    def to_dict(self):
        return asdict(self)


def analyze(symbol: str, current_price: float, prices: pd.DataFrame,
            models=('linear',), now=None) -> Decision:
    symbol = ticker(symbol)
    if not np.isfinite(current_price) or current_price <= 0:
        raise ValueError("Current price must be a positive finite number.")
    prices = validated(prices, symbol)
    now = pd.Timestamp(now or datetime.now(ZoneInfo('America/New_York')).date()).tz_localize(None).normalize()
    prices = prices.loc[prices.index < now]
    if len(prices) < 220:
        raise ValueError("Need at least 220 completed daily bars.")
    ind = indicator_pack(prices).iloc[-1]
    scores, detail, unavailable = components(prices, models)
    score = float(np.mean(list(scores.values())))
    side = action(score)
    warnings = ['Experimental rule/model agreement; not a calibrated success probability.']
    if unavailable:
        warnings.append('Some requested models were unavailable; result uses only listed components.')
    age = (now - prices.index[-1]).days
    gap = abs(current_price / prices.close.iloc[-1] - 1)
    if age > 5 or gap > max(.08, 3 * ind.atr14 / prices.close.iloc[-1]):
        side = 'WAIT'
        warnings.append('History is stale or the entered price is far from the last close. Refresh/verify before using levels.')
    if len(scores) < 2:
        warnings.append('Technical-only fallback; no ensemble was available.')
    recent_high = prices.high.iloc[-21:-1].max()
    recent_low = prices.low.iloc[-21:-1].min()
    closed_break = prices.close.iloc[-1] > recent_high or prices.close.iloc[-1] < recent_low
    breakout = 'Confirmed on daily close' if closed_break and ind.rvol20 >= 1.5 else (
        'Developing' if current_price > recent_high or current_price < recent_low else 'None')
    if ind.sweep_high or ind.sweep_low:
        breakout = 'Possible failure / liquidity sweep'
    quality = 'Good' if abs(score) >= .65 else 'Average' if side != 'WAIT' else 'Weak / wait'
    try:
        risk_levels = levels(current_price, float(ind.atr14), side)
    except ValueError as exc:
        side, quality = 'WAIT', 'Reject'
        warnings.append(str(exc))
        risk_levels = levels(current_price, float(ind.atr14), side)
    return Decision(symbol, float(current_price), str(prices.index[-1].date()),
                    'Bullish' if score >= .35 else 'Bearish' if score <= -.35 else 'Neutral',
                    side, quality, 'Possible' if ind.choch or ind.sweep_low or ind.sweep_high else 'Not detected',
                    breakout, **risk_levels, agreement_score=round(score, 4),
                    components=scores, model_details=detail, unavailable=unavailable, warnings=warnings)
