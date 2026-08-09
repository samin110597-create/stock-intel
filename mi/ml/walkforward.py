"""Walk-forward splitting with purge and embargo.

The subtlety that kills most retail backtests: with an h-day forward label,
training row t and test row t+1 overlap in the outcome they describe. A plain
train/test cut leaks the test period's future into training. Purging drops
the h rows before each test block; the embargo drops rows immediately after,
so the next fold's training set does not learn from the block just tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class Fold:
    index: int
    train: np.ndarray
    test: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    idx: pd.DatetimeIndex,
    n_folds: int = 6,
    horizon: int = 5,
    min_train: int = 250,
    embargo: int | None = None,
) -> Iterator[Fold]:
    n = len(idx)
    embargo = horizon if embargo is None else embargo
    usable = n - min_train
    if usable <= n_folds * 20:
        raise ValueError(f"{n} rows is too few for {n_folds} folds with min_train={min_train}")
    test_size = usable // n_folds

    for i in range(n_folds):
        test_start = min_train + i * test_size
        test_end = min_train + (i + 1) * test_size if i < n_folds - 1 else n
        train_end = test_start - horizon - embargo  # purge + embargo
        if train_end < min_train // 2:
            continue
        yield Fold(
            index=i,
            train=np.arange(0, train_end),
            test=np.arange(test_start, test_end),
            train_end=idx[train_end - 1],
            test_start=idx[test_start],
            test_end=idx[test_end - 1],
        )
