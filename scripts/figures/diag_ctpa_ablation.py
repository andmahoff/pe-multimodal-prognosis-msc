"""Stability check for the PE-descriptor ablation.
Repeats it across three CV seeds and adds a paired
subject-level bootstrap interval on the seed-42
predictions. Prints results only; no figure.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import fig_data as fd

PE_DESC = ["pe_pos", "pe_neg", "saddle", "central",
           "lobar", "segmental", "subseg", "bilateral",
           "rv_strain", "septal_bow", "reflux",
           "mpa_enlarge", "infarct"]
INCID = ["effusion", "malignancy", "consolid",
         "atelect", "edema", "cardiomeg", "adenopathy"]
META = ["txt_len", "n_sent"]

raw = pd.read_csv(fd.DATA + "/ctpa_comorb_features.csv")
cm = [c for c in raw.columns if c.startswith("cm_")]
dv = [c for c in raw.columns if c.startswith("dv_")]
WHITE = [c for c in PE_DESC + INCID + META + cm + dv
         if c in raw.columns]
RED = [c for c in WHITE if c not in PE_DESC]

pres = pd.read_csv(fd.DATA
                   + "/p_ctpa_pres_base_cm_dv_death_30d.csv")
keep = set(pres["hadm_id"])
lab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")
SEEDS = [42, 7, 13]
rng = np.random.RandomState(42)


def oof(d, y, g, cols, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                              random_state=seed)
    Xc = d[cols]
    for tr, te in cv.split(Xc, y, g):
        i2 = SimpleImputer(
            strategy="median").fit(Xc.iloc[tr])
        s2 = StandardScaler().fit(
            i2.transform(Xc.iloc[tr]))
        m = LogisticRegressionCV(Cs=10, cv=5,
                                 scoring="roc_auc",
                                 max_iter=5000,
                                 n_jobs=-1)
        m.fit(s2.transform(i2.transform(Xc.iloc[tr])),
              y[tr])
        p[te] = m.predict_proba(
            s2.transform(i2.transform(
                Xc.iloc[te])))[:, 1]
    return p


for o in ["death_30d", "composite_30d",
          "death_30d_inhosp"]:
    d = raw[raw["hadm_id"].isin(keep)][fd.KEY + WHITE]
    d = d.merge(lab[fd.KEY + [o]], on=fd.KEY).dropna(
        subset=[o])
    y = d[o].values.astype(float)
    g = d["subject_id"].values
    print("\n=== %s  n=%d ev=%d" % (o, len(d),
                                    int(y.sum())))
    deltas, p_full42, p_red42 = [], None, None
    for s in SEEDS:
        pf = oof(d, y, g, WHITE, s)
        pr = oof(d, y, g, RED, s)
        af, ar = roc_auc_score(y, pf), roc_auc_score(y, pr)
        deltas.append(ar - af)
        print("  seed %-4d full %.4f  minus PE %.4f  "
              "delta %+.4f" % (s, af, ar, ar - af))
        if s == 42:
            p_full42, p_red42 = pf, pr
    print("  MEAN delta %+.4f   range %+.4f to %+.4f"
          % (np.mean(deltas), min(deltas), max(deltas)))

    uniq = np.unique(g)
    idx = {q: np.where(g == q)[0] for q in uniq}
    bs = []
    for _ in range(2000):
        pick = rng.choice(uniq, size=len(uniq),
                          replace=True)
        r = np.concatenate([idx[q] for q in pick])
        if y[r].min() == y[r].max():
            continue
        bs.append(roc_auc_score(y[r], p_red42[r])
                  - roc_auc_score(y[r], p_full42[r]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("  seed-42 delta 95%% CI  [%+.4f, %+.4f]"
          % (lo, hi))

