"""Optional CPU model adapters. No hosted inference fees or remote execution."""
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _runtime():
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[2] / 'data' / 'models'))
    import torch
    torch.set_num_threads(max(1, int(os.getenv('ANALYZER_CPU_THREADS', '2'))))


QUESTIONS = {
    'direction': {'type': 'choice', 'instructions': 'Assess the five-session directional setup from these facts.',
                  'criteria': {'Bullish': 'Upward trend and momentum agree',
                               'Neutral': 'Mixed or inadequate evidence',
                               'Bearish': 'Downward trend and momentum agree'}},
    'action': {'type': 'choice', 'instructions': 'Which experimental action matches the evidence?',
               'criteria': {'LONG': 'Evidence supports upside', 'WAIT': 'Evidence is mixed or weak',
                            'SHORT': 'Evidence supports downside'}},
    'quality': {'type': 'score', 'instructions': 'Rate agreement and clarity of the setup, not profit probability.',
                'criteria': ['Reject', 'Weak', 'Average', 'Good', 'Excellent']},
    'reversal': {'type': 'noul', 'instructions': 'Do the supplied facts show a possible trend reversal?'},
    'breakout': {'type': 'choice', 'instructions': 'Is a price breakout supported by the supplied facts?',
                 'criteria': {'Confirmed': 'Price broke a prior range with high volume',
                              'Developing': 'Evidence suggests a forming breakout',
                              'None': 'No adequate breakout evidence'}},
}


@lru_cache(maxsize=1)
def _laya():
    _runtime()
    import laya
    return laya.load('convaiinnovations/laya-typed-decisions', device='cpu')


def laya_decisions(state, full=True):
    # Questions are separate batch rows. Historical trading only consumes
    # direction, so avoid evaluating four unused heads at every origin.
    result = _laya().predict(state, QUESTIONS if full else {'direction': QUESTIONS['direction']})
    answers = result['answers']
    choice = answers['direction']['choice']
    if choice not in ('Bullish', 'Neutral', 'Bearish'):
        raise ValueError('Laya returned an unknown direction')
    # Do not mistake out-of-domain model confidence for trade probability.
    score = {'Bullish': 1., 'Neutral': 0., 'Bearish': -1.}[choice]
    return score, {'answers': answers, 'checkpoint': 'convaiinnovations/laya-typed-decisions',
                   'status': 'Experimental; stock-domain probabilities are uncalibrated'}


@lru_cache(maxsize=1)
def _chronos():
    _runtime()
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu')


def chronos_score(prices, horizon=5):
    # A trading-session clock avoids weekends/holiday frequency irregularity.
    # Dates here are ordinal placeholders and carry no real calendar covariate.
    tail = prices.tail(256)
    context = pd.DataFrame({'id': 'stock',
                            'timestamp': pd.date_range('2000-01-01', periods=len(tail)),
                            'target': tail.close.to_numpy(),
                            'volume': np.log1p(tail.volume.to_numpy())})
    forecast = _chronos().predict_df(context, prediction_length=horizon,
                                     quantile_levels=[.1, .5, .9],
                                     id_column='id', timestamp_column='timestamp', target='target')
    median = float(forecast['0.5'].iloc[-1])
    expected = median / float(tail.close.iloc[-1]) - 1
    scale = max(float(tail.close.pct_change().std()) * np.sqrt(horizon), .005)
    score = float(np.tanh(expected / scale))
    if not np.isfinite(score) or median <= 0:
        raise ValueError('Chronos returned an invalid forecast')
    return score, {'median': median, 'low': float(forecast['0.1'].iloc[-1]),
                   'high': float(forecast['0.9'].iloc[-1]), 'checkpoint': 'amazon/chronos-2',
                   'status': 'Forecast interval is not a calibrated stock-price interval'}
