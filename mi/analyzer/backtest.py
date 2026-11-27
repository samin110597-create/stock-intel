"""Walk-forward ablations with next-open entry and non-overlapping positions.

Each model sees only a historical prefix. Missing models invalidate their
combinations instead of quietly substituting technical-only performance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mi.indicators import atr
from mi.redact import clean
from .data import load_history, validated
from .engine import action, components, levels

COMBINATIONS = {
    'technicals': ('technical',),
    'technicals+linear': ('technical', 'linear'),
    'technicals+chronos': ('technical', 'chronos'),
    'technicals+laya': ('technical', 'laya'),
    'technicals+chronos+laya': ('technical', 'chronos', 'laya'),
    'all': ('technical', 'linear', 'chronos', 'laya'),
}


def trade_path(bars, side, atr_value, cost_bps=15):
    """One unit of starting equity; stop-first if both barriers hit in a bar.

    Costs are total entry+exit commission/slippage. Short borrow is an extra
    3% annualized assumption. Cash earns zero. No pyramiding or leverage.
    """
    if side == 'WAIT':
        return np.ones(len(bars)), None
    entry = float(bars.open.iloc[0])
    plan = levels(entry, atr_value, side)
    stop, target = plan['stop'], plan['targets'][1]
    sign = 1 if side == 'LONG' else -1
    marks, exit_price, reason, exit_day = [], None, 'horizon', len(bars) - 1
    for j, row in enumerate(bars.itertuples()):
        if side == 'LONG':
            if row.open <= stop:
                exit_price, reason = row.open, 'gap stop'
            elif row.open >= target:
                exit_price, reason = target, 'target'
            elif row.low <= stop:
                exit_price, reason = stop, 'stop'
            elif row.high >= target:
                exit_price, reason = target, 'target'
        else:
            if row.open >= stop:
                exit_price, reason = row.open, 'gap stop'
            elif row.open <= target:
                exit_price, reason = target, 'target'
            elif row.high >= stop:
                exit_price, reason = stop, 'stop'
            elif row.low <= target:
                exit_price, reason = target, 'target'
        if j == len(bars) - 1 and exit_price is None:
            exit_price = row.close
        borrow = .03 * (j + 1) / 252 if sign < 0 else 0
        value = 1 + sign * ((exit_price if exit_price is not None else row.close) / entry - 1)
        value -= cost_bps / 10000 * (1 if exit_price is not None else .5) + borrow
        marks.append(float(value))
        if exit_price is not None:
            exit_day = j
            marks.extend([float(value)] * (len(bars) - j - 1))
            break
    if min(marks) <= 0:
        raise ValueError('Simulated equity exhausted; leveraged/short risk exceeds prototype assumptions.')
    return np.array(marks), {'entry': entry, 'exit': float(exit_price), 'stop': stop,
                              'target': target, 'exit_reason': reason,
                              'held_sessions': exit_day + 1, 'net_return': marks[-1] - 1,
                              'exit_date': str(bars.index[exit_day].date())}


def run(prices, models=('linear',), horizon=5, test_sessions=252, cost_bps=15, progress=None):
    prices = validated(prices, 'backtest', min_rows=520)
    if horizon < 1 or test_sessions < horizon or cost_bps < 0:
        raise ValueError('Invalid horizon, test period, or trading costs.')
    start = max(500, len(prices) - test_sessions - 1)
    origins = list(range(start, len(prices) - horizon, horizon))
    if not origins:
        raise ValueError('Not enough bars to evaluate the requested horizon.')
    records, paths, failures = [], {name: [1.] for name in COMBINATIONS}, {}
    disabled = {}
    for count, i in enumerate(origins):
        prefix = prices.iloc[:i + 1]
        scores, _, errors = components(prefix, tuple(m for m in models if m not in disabled), horizon, full_decisions=False)
        disabled.update(errors)
        errors = dict(disabled)
        a = float(atr(prefix).iloc[-1])
        future = prices.iloc[i + 1:i + horizon + 1]
        for name, required in COMBINATIONS.items():
            missing = set(required) - scores.keys()
            if missing:
                failures.setdefault(name, []).append({
                    'date': str(prefix.index[-1].date()),
                    'reason': {m: errors.get(m, 'not requested') for m in sorted(missing)}})
                continue
            score = float(np.mean([scores[k] for k in required]))
            side = action(score)
            try:
                path, trade = trade_path(future, side, a, cost_bps)
            except ValueError as exc:
                failures.setdefault(name, []).append({'date': str(prefix.index[-1].date()), 'reason': str(exc)})
                continue
            paths[name].extend(paths[name][-1] * path)
            records.append({'combination': name, 'signal_date': str(prefix.index[-1].date()),
                            'entry_date': str(future.index[0].date()), 'action': side, 'score': score,
                            'net_return': float(path[-1] - 1), 'trade': trade,
                            'horizon_price_return': float(future.close.iloc[-1] / future.open.iloc[0] - 1)})
        if progress:
            progress(f'{count + 1}/{len(origins)} historical decisions', flush=True)
    summary = {}
    for name in COMBINATIONS:
        if name in failures:
            summary[name] = {'status': 'unavailable', 'failed_origins': len(failures[name]),
                             'first_failure': failures[name][0]}
            continue
        rows = [r for r in records if r['combination'] == name]
        trades = [r for r in rows if r['trade']]
        equity = np.asarray(paths[name])
        summary[name] = {
            'status': 'evaluated', 'opportunities': len(rows), 'trades': len(trades),
            'net_return_pct': float((equity[-1] - 1) * 100),
            'max_drawdown_pct': float((equity / np.maximum.accumulate(equity) - 1).min() * 100),
            'win_rate_pct': float(np.mean([r['net_return'] > 0 for r in trades]) * 100) if trades else None,
            'direction_accuracy_pct': float(np.mean([
                (r['horizon_price_return'] > 0 if r['action'] == 'LONG' else r['horizon_price_return'] < 0)
                for r in trades]) * 100) if trades else None,
            'mean_net_return_per_opportunity_bps': float(np.mean([r['net_return'] for r in rows]) * 10000),
            'exposure_pct': float(sum(r['trade']['held_sessions'] for r in trades) / (len(rows) * horizon) * 100),
        }
    first, last = origins[0] + 1, origins[-1] + horizon
    buyhold = float((prices.close.iloc[last] / prices.open.iloc[first] - 1 - cost_bps / 10000) * 100)
    versions = {}
    for pkg in ('numpy', 'pandas', 'scikit-learn', 'laya', 'chronos-forecasting', 'torch', 'yfinance'):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = 'not installed'
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in Path(__file__).parent.glob('*.py')}
    import os
    model_home = Path(os.getenv('HF_HOME', str(Path(__file__).resolve().parents[2] / 'data' / 'models')))
    revisions = {path.parent.parent.name: path.read_text().strip()
                 for path in (model_home / 'hub').glob('models--*/refs/main')}
    return {
        'method': f'Expanding history; next-open entry; non-overlapping {horizon}-session slots; 1.5 ATR stop, 2R exit target; stop first on ambiguous bars.',
        'horizon_sessions': horizon, 'round_trip_cost_bps': cost_bps,
        'short_borrow_annual_pct': 3, 'test_start': str(prices.index[first].date()),
        'test_end': str(prices.index[last].date()), 'data_start': str(prices.index[0].date()),
        'data_rows': len(prices), 'data_sha256': hashlib.sha256(prices.to_csv().encode()).hexdigest(),
        'package_versions': versions, 'source_hashes': source_hashes, 'model_revisions': revisions,
        'buy_hold_price_return_pct': buyhold,
        'summary': summary, 'decisions': records,
        'caveats': [
            'Exploratory historical comparison, not proof of a reliable or profitable edge.',
            'User-selected surviving US tickers; survivorship and selection bias; no delisted universe.',
            'Foundation-model pretraining data may overlap test history; contamination cannot be excluded.',
            'No stock-domain probability calibration or model selection on an independent final holdout.',
            'Current vendor-adjusted price history, not archived point-in-time data; dividends and short dividend payments excluded.',
            'Daily OHLC cannot reconstruct intraday barrier ordering; conservative stop-first assumption.',
            'Fixed per-trade costs and borrow assumptions omit liquidity, locate availability, variable spreads and taxes.',
            'Drawdown uses daily closing marks; intraday drawdown can be worse. Entry uses next open, not a live quote.',
            'Full exit at target 2 or stop/horizon; target 1 and target 3 are reference levels, not partial fills.',
            'Lightweight/AI scores are experimental agreement scores. Legacy qualified-probability model is unchanged.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers', nargs='+', default=['NVDA', 'MU', 'SPY'])
    parser.add_argument('--start', default='2018-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--test-sessions', type=int, default=252)
    parser.add_argument('--models', default='linear', help='comma-separated: linear,chronos,laya')
    parser.add_argument('--cost-bps', type=float, default=15)
    parser.add_argument('--output', default='data/backtests/latest.json')
    parser.add_argument('--verify-laya', action='store_true', help='Compare full vs direction-only Laya batching on one input')
    args = parser.parse_args()
    report = {'results': {}, 'failures': {}}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    for symbol in args.tickers:
        print(f'Loading {symbol}', flush=True)
        try:
            data = load_history(symbol, args.start, args.end)
            if args.verify_laya and 'laya_batch_check' not in report:
                import time
                from .models import laya_decisions
                from .engine import state_for_laya, technical
                from mi.indicators import indicator_pack
                ind = indicator_pack(data.data).iloc[-1]
                state = state_for_laya(data.data, ind, {'technical': technical(ind, data.data.close.iloc[-1])})
                started = time.perf_counter()
                full_score, full = laya_decisions(state)
                full_seconds = time.perf_counter() - started
                started = time.perf_counter()
                narrow_score, narrow = laya_decisions(state, full=False)
                narrow_seconds = time.perf_counter() - started
                if full_score != narrow_score:
                    raise ValueError('Laya full and narrow batches disagree on direction; aborting verification')
                report['laya_batch_check'] = {'same_direction': True, 'full_answers': full['answers'],
                                              'direction_only': narrow['answers'],
                                              'full_seconds_including_load': full_seconds,
                                              'direction_only_seconds': narrow_seconds}
                print('Laya full/narrow direction agreement verified', flush=True)
            result = run(data.data, tuple(filter(None, args.models.split(','))),
                         test_sessions=args.test_sessions, cost_bps=args.cost_bps, progress=print)
            result['source'] = data.sources()
            result['fetched_at'] = data.provenance['close'].fetched_at.isoformat()
            report['results'][symbol] = result
        except Exception as exc:
            report['failures'][symbol] = str(clean(f'{type(exc).__name__}: {exc}'))
            print(f'{symbol}: unavailable', flush=True)
        target.write_text(json.dumps(clean(report), indent=2, allow_nan=False), encoding='utf-8')
    print(f'Report written: {target}', flush=True)
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
