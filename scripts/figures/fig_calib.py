"""Out-of-fold Platt calibration, shared by the
calibration, decision-curve and reclassification
figures. Uses the same grouped 5-fold scheme as every
other analysis, so no patient is calibrated on their
own outcome.

Platt scaling is fitted on the log-odds of each score.
Fitting it on the probability itself makes the
calibrated range depend on the input range, which
compresses any modality whose raw scores are narrow.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

EPS = 1e-6


def to_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1.0 - p))


def platt_oof(y, p, groups, seed=42):
    y = np.asarray(y, dtype=float)
    x = to_logit(p).reshape(-1, 1)
    out = np.zeros(len(y), dtype=float)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                              random_state=seed)
    for tr, te in cv.split(x, y, groups):
        lr = LogisticRegression(C=1e6, max_iter=2000)
        lr.fit(x[tr], y[tr])
        out[te] = lr.predict_proba(x[te])[:, 1]
    return out


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def net_benefit(y, p, thr):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = float(len(y))
    pos = p >= thr
    tp = float(np.sum(pos & (y == 1)))
    fp = float(np.sum(pos & (y == 0)))
    return tp / n - (fp / n) * (thr / (1.0 - thr))

