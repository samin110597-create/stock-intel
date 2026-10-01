import numpy as np
import pandas as pd
import pytest

from mi.analyzer.data import validated
from mi.analyzer.engine import analyze, features, levels, linear_score
from mi.analyzer.backtest import run, trade_path
from tests.conftest import make_ohlcv


@pytest.fixture
def prices():
    p = make_ohlcv(620)
    p['high'] = p[['open', 'high']].max(axis=1)
    p['low'] = p[['open', 'low']].min(axis=1)
    return p


def test_prefix_features_are_causal(prices):
    pd.testing.assert_frame_equal(features(prices).iloc[:550], features(prices.iloc[:550]))


def test_rsi_no_losses_is_overbought():
    from mi.indicators import rsi
    assert rsi(pd.Series(range(100), dtype=float)).iloc[-1] == 100
    assert rsi(pd.Series([100.] * 100)).iloc[-1] == 50


def test_linear_model_never_receives_future(prices, monkeypatch):
    import mi.analyzer.backtest as bt
    observed = []
    def inspect(prefix, models, horizon, **kwargs):
        observed.append(prefix.index[-1])
        return {'technical': .5, 'linear': .4}, {}, {}
    monkeypatch.setattr(bt, 'components', inspect)
    result = run(prices, test_sessions=20)
    firsts = [r for r in result['decisions'] if r['combination'] == 'technicals']
    assert len(observed) == len(firsts)
    assert all(pd.Timestamp(r['signal_date']) < pd.Timestamp(r['entry_date']) for r in firsts)
    assert observed == [pd.Timestamp(r['signal_date']) for r in firsts]
    changed = prices.copy()
    changed.iloc[550:, changed.columns.get_loc('close')] *= 2
    assert linear_score(prices.iloc[:550]) == linear_score(changed.iloc[:550])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 0])
def test_bad_price_rejected(prices, value):
    with pytest.raises(ValueError):
        analyze('NVDA', value, prices)


def test_rejects_partial_ohlcv(prices):
    prices.loc[prices.index[-1], 'volume'] = np.nan
    with pytest.raises(ValueError):
        validated(prices, 'NVDA')


def test_wait_for_stale_or_mismatched_quote(prices):
    r = analyze('NVDA', float(prices.close.iloc[-1]), prices, models=(), now='2030-01-01')
    assert r.action == 'WAIT' and r.entry is None and not r.targets
    r = analyze('NVDA', 10000, prices, models=(), now=prices.index[-1] + pd.Timedelta(days=1))
    assert r.action == 'WAIT' and r.stop is None


def test_missing_models_are_not_fake_ablations(prices):
    result = run(prices, models=(), test_sessions=10)
    assert result['summary']['technicals']['status'] == 'evaluated'
    assert result['summary']['technicals+laya']['status'] == 'unavailable'
    assert 'net_return_pct' not in result['summary']['technicals+laya']


def test_current_day_excluded(prices):
    now = prices.index[-1]
    r = analyze('NVDA', float(prices.close.iloc[-2]), prices, models=(), now=now)
    assert r.as_of == str(prices.index[-2].date())


def test_short_level_order():
    p = levels(100, 2, 'SHORT')
    assert p['stop'] > p['entry'] > p['targets'][0] > p['targets'][1] > p['targets'][2] > 0


def test_stop_first_and_costs():
    bars = pd.DataFrame({'open': [100.], 'high': [110.], 'low': [90.], 'close': [105.]},
                        index=pd.date_range('2025-01-01', periods=1))
    path, trade = trade_path(bars, 'LONG', 2, 15)
    assert trade['exit'] == 97
    assert path[-1] == pytest.approx(.9685)


def test_gap_stop_fills_at_open():
    bars = pd.DataFrame({'open': [100., 90.], 'high': [101., 92.], 'low': [99., 89.], 'close': [100., 91.]},
                        index=pd.date_range('2025-01-01', periods=2))
    _, trade = trade_path(bars, 'LONG', 2)
    assert trade['exit'] == 90 and trade['exit_reason'] == 'gap stop'


def test_zero_trades_is_defined(prices, monkeypatch):
    import mi.analyzer.backtest as bt
    monkeypatch.setattr(bt, 'components', lambda *a, **kw: ({'technical': 0}, {}, {}))
    row = run(prices, models=(), test_sessions=10)['summary']['technicals']
    assert row['trades'] == 0 and row['net_return_pct'] == 0 and row['win_rate_pct'] is None


def test_ui_success_and_failure(prices, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from mi.provenance import uniform_provenance
    import mi.analyzer.data as data
    monkeypatch.setattr(data, 'load_history', lambda *a: uniform_provenance(prices, 'test', 'test'))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'analyzer_app.py', default_timeout=30).run()
    assert len(app.text_input) == 1 and len(app.number_input) == 1
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) >= 3
    app.text_input[0].set_value('<invalid>')
    app.button[0].click().run(timeout=30)
    assert not app.exception and len(app.error) == 1
