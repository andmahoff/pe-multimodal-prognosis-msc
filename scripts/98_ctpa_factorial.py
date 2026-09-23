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
EXT = (BASE + "/mimic_ext_pe/physionet.org/files"
       "/mimic-iv-ext-pe/1.0.0/MIMIC-IV-Ext-PE.csv")
NOTES = DATA + "/ctpa_notes_index.csv"
LAB = DATA + "/mimic_labels_harmonised.csv"
OUT = DATA + "/ctpa_factorial_results.csv"

SEEDS = [42, 7, 13]
OUTS = ["cv_first", "composite_30d",
        "death_30d", "death_30d_inhosp"]

NEG = (r"\bno\b|\bnot\b|\bwithout\b|\bnone\b"
       r"|\bnor\b|\bnegative for\b|\babsent\b"
       r"|\bfree of\b|\bunremarkable\b"
       r"|\bresolved\b|\bruled out\b")

RULES = {
    "pe_pos": r"pulmonary embol|filling defect",
    "saddle": r"saddle",
    "central": r"\bcentral\b|main pulmonary",
    "lobar": r"lobar",
    "segmental": r"(?<!sub)segmental",
    "subseg": r"subsegmental",
    "bilateral": r"bilateral",
    "rv_strain": r"right ventric|rv[/ ]lv"
                 r"|right heart strain|rv strain",
    "septal_bow": r"septal bowing"
                  r"|septal flattening|d-?shaped",
    "reflux": r"reflux[^.]{0,25}"
              r"(?:ivc|vena cava|hepatic)"
              r"|reflux of contrast",
    "mpa_enlarge": r"pulmonary arter\w+"
                   r"[^.]{0,30}(?:enlarg|dilat)",
    "infarct": r"infarct",
    "effusion": r"effusion",
    "malignancy": r"malignan|metasta|neoplas"
                  r"|carcinoma|\bmass\b",
    "consolid": r"consolidat|pneumonia",
    "atelect": r"atelecta",
    "edema": r"edema",
    "cardiomeg": r"cardiomegal",
    "adenopathy": r"adenopathy",
}

NUM = {
    "rvlv": [r"rv\s*[/:]\s*lv[^0-9]{0,20}"
             r"(\d+(?:\.\d+)?)",
             r"right[^.]{0,12}left[^.]{0,12}"
             r"ratio[^0-9]{0,12}"
             r"(\d+(?:\.\d+)?)"],
    "paao": [r"p[au]\s*[/:]\s*a[o]?"
             r"[^0-9]{0,15}(\d+(?:\.\d+)?)"],
    "mpa_cm": [r"(?:main )?pulmonary arter\w+"
               r"[^.]{0,40}?(\d+(?:\.\d+)?)\s*cm"],
}
NUMRANGE = {"rvlv": (0.2, 5.0),
            "paao": (0.2, 3.0),
            "mpa_cm": (1.0, 6.0)}


def sect(t, start, ends):
    m = re.search(start, t, flags=re.I)
    if not m:
        return ""
    s = t[m.end():]
    cut = re.search(ends, s, flags=re.I)
    if cut:
        s = s[:cut.start()]
    s = re.sub(r"_{2,}", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def get_imp(t):
    return sect(str(t), r"IMPRESSION[S]?\s*:",
                r"\n\s*(?:ADDENDUM|NOTIFICATION"
                r"|WET READ|PRELIMINARY)")


def get_find(t):
    return sect(str(t), r"FINDINGS?\s*:",
                r"IMPRESSION[S]?\s*:")


def clauses(s):
    s = re.sub(r"(\d)\.(\d)", r"\1<D>\2", s)
    ps = re.split(r"[.;:\n]|\b(?:but|however"
                  r"|although)\b", s)
    return [p.replace("<D>", ".").strip()
            for p in ps if p.strip()]


def hit(cs, pat, want_neg=False):
    for c in cs:
        m = re.search(pat, c)
        if not m:
            continue
        neg = bool(re.search(NEG, c[:m.start()]))
        if neg == want_neg:
            return 1
    return 0


def num(s, key):
    lo, hi = NUMRANGE[key]
    vals = []
    for p in NUM[key]:
        for m in re.finditer(p, s):
            try:
                v = float(m.group(1))
            except Exception:
                continue
            if lo <= v <= hi:
                vals.append(v)
    return max(vals) if vals else np.nan


def featurise(txt):
    low = txt.lower()
    cs = clauses(low)
    d = {}
    for k, pat in RULES.items():
        d[k] = hit(cs, pat)
    d["pe_neg"] = hit(cs, RULES["pe_pos"], True)
    for k in NUM:
        v = num(low, k)
        d[k + "_val"] = v
        d[k + "_pres"] = 0 if np.isnan(v) else 1
    d["txt_len"] = len(txt)
    d["n_sent"] = len(cs)
    return d


lab = pd.read_csv(LAB)
nt = pd.read_csv(NOTES)
nt["charttime"] = pd.to_datetime(
    nt["charttime"], errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm", as_index=False).first()
print("first-scan admissions:", len(nt))

ext = pd.read_csv(EXT, usecols=["note_id",
                                "Type_of_PE"])
ext["Type_of_PE"] = (ext["Type_of_PE"]
                     .astype(str).str.strip()
                     .str.lower())
print("")
print("Type_of_PE (normalised):")
print(ext["Type_of_PE"].value_counts())

nt = nt.merge(ext, on="note_id", how="left")
mt = nt["Type_of_PE"].notna()
print("")
print("matched to Ext-PE:", int(mt.sum()),
      "of", len(nt))
nt["Type_of_PE"] = nt["Type_of_PE"].fillna(
    "not_in_extpe")
print("")
print("label distribution in cohort:")
print(nt["Type_of_PE"].value_counts())

nt["imp"] = nt["text"].apply(get_imp)
nt["fnd"] = nt["text"].apply(get_find)
print("")
print("with impression:",
      int((nt["imp"].str.len() > 0).sum()))
print("with findings:  ",
      int((nt["fnd"].str.len() > 0).sum()))
print("median imp chars:",
      int(nt["imp"].str.len().median()))
print("median fnd chars:",
      int(nt["fnd"].str.len().median()))

nt["both"] = (nt["fnd"] + " "
              + nt["imp"]).str.strip()

pe1 = pd.get_dummies(nt["Type_of_PE"],
                     prefix="ext").astype(int)
EXTC = list(pe1.columns)
print("ext dummy cols:", EXTC)

BASECOLS = None
FR = {}
for src in ["imp", "both"]:
    rows = [featurise(t) for t in nt[src]]
    fx = pd.DataFrame(rows)
    if BASECOLS is None:
        BASECOLS = list(fx.columns)
    print("")
    print("[%s] rvlv_pres %.3f  paao_pres %.3f"
          "  mpa_cm_pres %.3f"
          % (src, fx["rvlv_pres"].mean(),
             fx["paao_pres"].mean(),
             fx["mpa_cm_pres"].mean()))
    print("[%s] rv_strain %.3f  septal %.3f"
          "  reflux %.3f"
          % (src, fx["rv_strain"].mean(),
             fx["septal_bow"].mean(),
             fx["reflux"].mean()))
    fx.columns = [src + "__" + c
                  for c in fx.columns]
    FR[src] = fx.reset_index(drop=True)

allfx = pd.concat(
    [FR["imp"], FR["both"],
     pe1.reset_index(drop=True)], axis=1)
allfx["hadm_id"] = nt["idx_hadm"].values
d = lab.merge(allfx, on="hadm_id", how="inner")
print("")
print("merged:", len(d))
d.to_csv(DATA + "/ctpa_factorial_features.csv",
         index=False)


def cols_for(src, pelab):
    cs = [src + "__" + c for c in BASECOLS]
    if pelab == "ext":
        cs = [c for c in cs
              if not c.endswith("__pe_pos")
              and not c.endswith("__pe_neg")]
        cs = cs + EXTC
    return cs


def make_model():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=10, cv=3, scoring="roc_auc",
            max_iter=5000, n_jobs=-1))])


res = []
for out in OUTS:
    dd = d
    if out == "cv_first":
        dd = d[d["death_first"] == 0]
    y = dd[out].values.astype(int)
    grp = dd["subject_id"].values
    if y.sum() < 20:
        continue
    print("")
    print("=" * 56)
    print(out, "n=%d ev=%d (%.4f)"
          % (len(y), int(y.sum()), y.mean()))

    for src in ["imp", "both"]:
        for pelab in ["regex", "ext"]:
            XC = [c for c in cols_for(src, pelab)
                  if c in dd.columns]
            X = dd[XC].values.astype(float)
            keep = ~np.all(np.isnan(X), axis=0)
            X = X[:, keep]
            aucs = []
            for s in SEEDS:
                cv = StratifiedGroupKFold(
                    n_splits=5, shuffle=True,
                    random_state=s)
                oof = np.zeros(len(y))
                for tr, te in cv.split(X, y, grp):
                    md = make_model()
                    md.fit(X[tr], y[tr])
                    oof[te] = md.predict_proba(
                        X[te])[:, 1]
                aucs.append(roc_auc_score(y, oof))
                if s == 42:
                    o42 = oof.copy()
                    ap = average_precision_score(
                        y, oof)
            mu = float(np.mean(aucs))
            sd = float(np.std(aucs))
            tag = src + "_" + pelab
            print("  %-11s AUC %.4f +/- %.4f "
                  "AP %.4f  (%d feats)"
                  % (tag, mu, sd, ap, X.shape[1]))
            res.append({"outcome": out,
                        "text": src,
                        "pelabel": pelab,
                        "n": len(y),
                        "ev": int(y.sum()),
                        "auc": mu, "sd": sd,
                        "ap": ap,
                        "nfeat": X.shape[1]})
            po = pd.DataFrame({
                "subject_id":
                    dd["subject_id"].values,
                "hadm_id": dd["hadm_id"].values,
                "p_ctpa": o42})
            po.to_csv(DATA + "/p_ctpa_" + tag
                      + "_" + out + ".csv",
                      index=False)

rdf = pd.DataFrame(res)
rdf.to_csv(OUT, index=False)
print("")
print(rdf.round(4).to_string())

print("")
print("EFFECT DECOMPOSITION (AUC deltas)")
for out in rdf["outcome"].unique():
    s = rdf[rdf["outcome"] == out]

    def g(t, p):
        v = s[(s["text"] == t)
              & (s["pelabel"] == p)]["auc"]
        return float(v.iloc[0]) if len(v) else np.nan

    a = g("imp", "regex")
    b = g("imp", "ext")
    c = g("both", "regex")
    e = g("both", "ext")
    print("")
    print(out)
    print("  ExtPE effect: imp %+.4f"
          "  findings %+.4f" % (b - a, e - c))
    print("  Findings eff: regex %+.4f"
          "  extPE %+.4f" % (c - a, e - b))
    print("  best %.4f" % np.nanmax([a, b, c, e]))
print("saved", OUT)
