import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
OUT = DATA + "/count_feature_results.csv"
SEEDS = [42, 7, 13]

X = np.load(DATA + "/inspect_count_X.npy")
coh = pd.read_csv(
    DATA + "/inspect_count_cohort.csv")
y = coh["y"].values.astype(int)
grp = coh["person_id"].values
print("X", X.shape, "ev", int(y.sum()))
Xl = np.log1p(X)


def evalcv(Xm, mk):
    aucs = []
    ap = np.nan
    for sd in SEEDS:
        cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True,
            random_state=sd)
        oof = np.zeros(len(y))
        for tr, te in cv.split(Xm, y, grp):
            m = mk()
            m.fit(Xm[tr], y[tr])
            oof[te] = m.predict_proba(
                Xm[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
        if sd == 42:
            ap = average_precision_score(y, oof)
    return (float(np.mean(aucs)),
            float(np.std(aucs)), ap)


def mk_gb():
    return HistGradientBoostingClassifier(
        random_state=42, max_depth=3,
        learning_rate=0.05, max_iter=300,
        l2_regularization=1.0)


def mk_lr():
    return Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.01, max_iter=3000))])


res = []
for nm, Xm, mk in [("counts_gb", X, mk_gb),
                   ("log1p_gb", Xl, mk_gb),
                   ("log1p_lr", Xl, mk_lr)]:
    a, sd, ap = evalcv(Xm, mk)
    print("")
    print("%-11s AUC %.4f +/- %.4f  AP %.4f"
          % (nm, a, sd, ap))
    tr = (coh["split"].astype(str)
          == "train").values
    te = coh["split"].astype(str).isin(
        ["test", "valid"]).values
    off = np.nan
    if y[te].sum() >= 10:
        m = mk()
        m.fit(Xm[tr], y[tr])
        off = roc_auc_score(
            y[te], m.predict_proba(Xm[te])[:, 1])
        print("   official split %.4f" % off)
    res.append({"model": nm, "n": len(y),
                "ev": int(y.sum()),
                "nfeat": Xm.shape[1],
                "auc": a, "sd": sd, "ap": ap,
                "official": off})

pd.DataFrame(res).to_csv(OUT, index=False)
print("")
print(pd.DataFrame(res).round(4).to_string())
print("")
print("REF 15-feature EHR modality 0.6976")
print("REF paper LightGBM         0.848")
print("REF paper MOTOR            0.923")

