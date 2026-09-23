import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
NOTES = DATA + "/ctpa_notes_index.csv"
LAB = DATA + "/mimic_labels_harmonised.csv"
FEAT = DATA + "/ctpa_routeA_features.csv"

SEEDS = [42, 7, 13]
OUTS = ["cv_first", "composite_30d",
        "death_30d", "death_30d_inhosp"]

NEG = (r"\bno\b|\bnot\b|\bwithout\b|\bnone\b"
       r"|\bnor\b|\bnegative for\b|\babsent\b"
       r"|\bfree of\b|\bunremarkable\b"
       r"|\bresolved\b|\bresolution of\b"
       r"|\bruled out\b|\bexcluded\b")

HEDGE = (r"possibl|cannot be excluded"
         r"|cannot exclude|suboptimal"
         r"|limited (?:study|evaluation|by)"
         r"|equivocal|indeterminate"
         r"|suspicious for|may represent"
         r"|difficult to|degraded")

URGENT = (r"communicated|notified"
          r"|discussed with|telephone"
          r"|critical (?:result|finding)"
          r"|wet read")

RULES = {
    "pe_pos": r"pulmonary embol|filling defect"
              r"|thrombus in the pulmonary",
    "saddle": r"saddle",
    "central": r"\bcentral\b|main pulmonary"
               r"|bilateral main",
    "lobar": r"lobar",
    "segmental": r"(?<!sub)segmental",
    "subseg": r"subsegmental",
    "bilateral": r"bilateral",
    "occlusive": r"(?<!non-)(?<!non)occlusive",
    "rv_strain": r"right ventric|rv[/ ]lv"
                 r"|right heart strain|rv strain"
                 r"|right ventricular strain",
    "septal_bow": r"septal bowing"
                  r"|septal flattening"
                  r"|flatten\w*[^.]{0,20}septum"
                  r"|d-?shaped",
    "reflux": r"reflux of contrast"
              r"|contrast reflux"
              r"|reflux[^.]{0,25}"
              r"(?:ivc|vena cava|hepatic)",
    "mpa_enlarge": r"(?:main )?pulmonary"
                   r" arter\w+[^.]{0,30}"
                   r"(?:enlarg|dilat)"
                   r"|enlarged main pulmonary",
    "infarct": r"infarct",
    "effusion": r"effusion",
    "malignancy": r"malignan|metasta|neoplas"
                  r"|carcinoma|\bmass\b",
    "consolid": r"consolidat|pneumonia",
    "atelect": r"atelecta",
    "edema": r"edema",
    "cardiomeg": r"cardiomegal|enlarged heart",
    "chronic_pe": r"chronic[^.]{0,30}embol",
    "adenopathy": r"adenopathy|lymphadenopathy",
}

RVLV = [
    r"rv\s*[/:]\s*lv[^0-9]{0,20}"
    r"(\d+(?:\.\d+)?)",
    r"right[^.]{0,15}left[^.]{0,15}ratio"
    r"[^0-9]{0,15}(\d+(?:\.\d+)?)",
]


def impression(t):
    t = str(t)
    m = re.search(r"IMPRESSION[S]?\s*:", t,
                  flags=re.I)
    if not m:
        return ""
    s = t[m.end():]
    cut = re.search(
        r"\n\s*(?:ADDENDUM|NOTIFICATION"
        r"|WET READ|PRELIMINARY)", s,
        flags=re.I)
    if cut:
        s = s[:cut.start()]
    s = re.sub(r"_{2,}", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clauses(s):
    s = re.sub(r"(\d)\.(\d)", r"\1<D>\2", s)
    ps = re.split(
        r"[.;:\n]|\b(?:but|however"
        r"|although|whereas)\b", s)
    out = []
    for p in ps:
        p = p.replace("<D>", ".").strip()
        if p:
            out.append(p)
    return out


def hit(cs, pat, want_neg=False):
    for c in cs:
        m = re.search(pat, c)
        if not m:
            continue
        pre = c[:m.start()]
        neg = bool(re.search(NEG, pre))
        if neg == want_neg:
            return 1
    return 0


def rvlv(s):
    vals = []
    for p in RVLV:
        for m in re.finditer(p, s):
            try:
                v = float(m.group(1))
            except Exception:
                continue
            if 0.2 <= v <= 5.0:
                vals.append(v)
    if not vals:
        return np.nan
    return max(vals)


def featurise(imp):
    low = imp.lower()
    cs = clauses(low)
    d = {}
    for k, pat in RULES.items():
        d[k] = hit(cs, pat)
    d["pe_neg"] = hit(cs, RULES["pe_pos"],
                      want_neg=True)
    d["hedge"] = hit(cs, HEDGE)
    d["urgent"] = hit(cs, URGENT)
    v = rvlv(low)
    d["rvlv_present"] = 0 if np.isnan(v) else 1
    d["rvlv_value"] = v
    d["imp_len"] = len(imp)
    d["n_sent"] = len(cs)
    return d


lab = pd.read_csv(LAB)
nt = pd.read_csv(NOTES)
print("notes:", len(nt))

nt["charttime"] = pd.to_datetime(
    nt["charttime"], errors="coerce")
nt["imp"] = nt["text"].apply(impression)
nt = nt[nt["imp"].str.len() > 0].copy()
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm", as_index=False).first()
print("first-scan admissions:", len(nt))

rows = [featurise(t) for t in nt["imp"]]
fx = pd.DataFrame(rows)
fx["hadm_id"] = nt["idx_hadm"].values

XC = list(fx.columns)
XC.remove("hadm_id")

d = lab.merge(fx, on="hadm_id", how="inner")
print("merged:", len(d))
d.to_csv(FEAT, index=False)

BIN = [c for c in XC
       if c not in ("rvlv_value", "imp_len",
                    "n_sent")]
print("")
print("feature prevalence (negation-aware):")
print(d[BIN].mean().round(3))
print("")
print("rvlv_value non-null:",
      int(d["rvlv_value"].notna().sum()),
      "median:",
      round(float(d["rvlv_value"].median()), 3)
      if d["rvlv_value"].notna().any() else "na")


def make_model():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=10, cv=3, scoring="roc_auc",
            max_iter=5000, n_jobs=-1))])


def run(X, y, grp, seeds):
    aucs = []
    o42 = None
    ap42 = np.nan
    for s in seeds:
        cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True,
            random_state=s)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, grp):
            m = make_model()
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
        if s == 42:
            o42 = oof.copy()
            ap42 = average_precision_score(y, oof)
    return aucs, o42, ap42


res = []
for out in OUTS:
    if out not in d.columns:
        continue
    dd = d
    if out == "cv_first":
        dd = d[d["death_first"] == 0]
    y = dd[out].values.astype(int)
    X = dd[XC].values.astype(float)
    grp = dd["subject_id"].values
    if y.sum() < 20:
        continue

    aucs, o42, ap42 = run(X, y, grp, SEEDS)
    mu = float(np.mean(aucs))
    sd = float(np.std(aucs))
    print("")
    print("=" * 50)
    print(out, "n=%d ev=%d (%.4f)"
          % (len(y), int(y.sum()), y.mean()))
    print("  AUC %.4f +/- %.4f  AP %.4f"
          % (mu, sd, ap42))
    res.append({"outcome": out, "n": len(y),
                "ev": int(y.sum()), "auc": mu,
                "sd": sd, "ap": ap42})

    po = pd.DataFrame({
        "subject_id": dd["subject_id"].values,
        "hadm_id": dd["hadm_id"].values,
        "p_ctpa": o42})
    po.to_csv(DATA + "/p_ctpa_routeA_"
              + out + ".csv", index=False)

    m = make_model()
    m.fit(X, y)
    s = pd.Series(m.named_steps["lr"].coef_[0],
                  index=XC).sort_values(
        ascending=False)
    print("  top +:", s.head(6).round(3).to_dict())
    print("  top -:", s.tail(5).round(3).to_dict())
    key = ["rv_strain", "septal_bow",
           "rvlv_value", "reflux", "pe_pos"]
    print("  KEY:",
          {k: round(float(s[k]), 3)
           for k in key if k in s})

    mk = dd["pe_pos"] == 1
    y2 = y[mk.values]
    X2 = X[mk.values]
    g2 = grp[mk.values]
    if y2.sum() >= 20:
        a2, _, _ = run(X2, y2, g2, [42])
        print("  pe_pos only: n=%d ev=%d AUC %.4f"
              % (len(y2), int(y2.sum()), a2[0]))

rdf = pd.DataFrame(res)
rdf.to_csv(DATA + "/ctpa_routeA_results.csv",
           index=False)
print("")
print(rdf.to_string())
