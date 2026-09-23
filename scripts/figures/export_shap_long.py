"""Exports per-patient SHAP values for the EHR modality
so the beeswarm figures can be drawn in R.

phi_ij = beta_j * (z_ij - mean_target(z_j))

cv_first is a competing-risk outcome: patients who died
before any cardiovascular readmission are not at risk
and must leave the denominator on both sides. The
exclusion rule matches scripts 126, 127 and 129, which
use
    d[d["death_first"] == 0]
on the INSPECT and MIMIC label frames alike.

fig_feats.load_pair does not apply that exclusion, so
this script points fd.DATA at a temporary directory
holding filtered copies of the two label files plus
symbolic links to everything else. fig_feats itself is
not modified. Creating symbolic links on Windows needs
Developer Mode or administrator rights; the script was
run on Linux.
"""
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import fig_feats as ff

OUT = "./fusion_workspace/tables"

OUTCOMES = ["death_30d", "composite_30d", "cv_first"]
AT_RISK = ["cv_first"]

SRC_LAB = "inspect_labels_final.csv"
TGT_LAB = "mimic_labels_harmonised.csv"
EXCL = "death_first"


def mirror_dir(tmp):
    """Symlink every file in fd.DATA into tmp. The
    links need absolute targets, because fd.DATA is a
    relative path and a relative link would resolve
    inside tmp."""
    src = os.path.abspath(ff.fd.DATA)
    for f in os.listdir(src):
        s = os.path.join(src, f)
        if os.path.isfile(s):
            os.symlink(s, os.path.join(tmp, f))


def drop_died_first(tmp, name):
    """Swap one label file for its at-risk subset,
    using the same == 0 rule as scripts 126/127/129."""
    orig = os.path.join(ff.fd.DATA, name)
    link = os.path.join(tmp, name)
    d = pd.read_csv(orig)
    if EXCL not in d.columns:
        raise ValueError("no %s in %s" % (EXCL, name))
    n_na = int(d[EXCL].isna().sum())
    if n_na:
        raise ValueError(
            "%d null %s in %s - resolve before "
            "filtering" % (n_na, EXCL, name))
    n0 = len(d)
    d = d[d[EXCL] == 0].copy()
    print("  %s: %d -> %d (dropped %d)"
          % (name, n0, len(d), n0 - len(d)))
    # remove the link first, so the filtered copy is
    # written into tmp rather than over the original
    os.remove(link)
    d.to_csv(link, index=False)


def load(outcome):
    if outcome not in AT_RISK:
        return ff.load_pair(outcome)
    print("  applying competing-risk exclusion")
    keep = ff.fd.DATA
    tmp = tempfile.mkdtemp(prefix="atrisk_")
    try:
        mirror_dir(tmp)
        drop_died_first(tmp, SRC_LAB)
        drop_died_first(tmp, TGT_LAB)
        ff.fd.DATA = tmp
        return ff.load_pair(outcome)
    finally:
        ff.fd.DATA = keep
        shutil.rmtree(tmp, ignore_errors=True)


for o in OUTCOMES:
    print("=" * 50)
    print("outcome:", o)

    Xs, ys, Xt, yt, cols = load(o)

    print("source rows:", Xs.shape[0],
          "events:", int(np.sum(ys)))
    print("target rows:", Xt.shape[0],
          "events:", int(np.sum(yt)))
    print("features:", len(cols))

    imp = SimpleImputer(strategy="median").fit(Xs)
    sc = StandardScaler().fit(imp.transform(Xs))
    zs = sc.transform(imp.transform(Xs))
    zt = sc.transform(imp.transform(Xt))

    lr = LogisticRegression(max_iter=5000).fit(zs, ys)
    beta = lr.coef_[0]

    p = lr.predict_proba(zt)[:, 1]
    print("refit target AUC: %.4f"
          % roc_auc_score(yt, p))

    shap = beta * (zt - zt.mean(axis=0))

    d = pd.DataFrame(shap, columns=cols)
    d = d.melt(var_name="feature", value_name="shap")
    v = pd.DataFrame(zt, columns=cols).melt(
        var_name="feature", value_name="zval")
    d["zval"] = v["zval"]

    f = OUT + "/shap_long_%s_target.csv" % o
    d.to_csv(f, index=False)
    print("wrote", f, d.shape)

    b = pd.DataFrame({"feature": cols, "beta": beta})
    b = b.reindex(
        b.beta.abs().sort_values(ascending=False).index)
    b.to_csv(OUT + "/shap_beta_%s.csv" % o, index=False)
    print(b.head(8).to_string(index=False))