import re
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
INS = BASE + "/inspect_files"
DATA = BASE + "/fusion_workspace/data"
D = "_20250611.tsv"
NOTES = DATA + "/ctpa_notes_index.csv"
LAB = DATA + "/mimic_labels_harmonised.csv"
MOUT = DATA + "/mimic_imp_features_v2.csv"

SEEDS = [42, 7, 13]
GAP_D = 30
POST_H = 24

NEG = (r"\bno\b|\bnot\b|\bwithout\b|\bnone\b"
       r"|\bnor\b|\bnegative for\b|\babsent\b"
       r"|\bfree of\b|\bunremarkable\b"
       r"|\bresolved\b|\bruled out\b")

BASER = {
    "pe_pos": r"pulmonary embol|filling defect",
    "saddle": r"saddle",
    "central": r"\bcentral\b(?!\s*(?:line|venous"
               r"|cath|access))|main pulmonary",
    "lobar": r"lobar",
    "segmental": r"(?<!sub)segmental",
    "subseg": r"subsegmental",
    "bilateral": r"bilateral",
    "rv_strain": r"right ventric|rv[/ ]lv"
                 r"|right heart strain|rv strain",
    "septal_bow": r"septal bowing"
                  r"|septal flattening|d-?shaped",
    "reflux": r"reflux[^.]{0,25}"
              r"(?:ivc|vena cava|hepatic)",
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

COMORB = {
    "emphysema": r"emphysem|\bcopd\b"
                 r"|centrilobular",
    "fibrosis": r"fibrosis|interstitial"
                r"|honeycomb|reticulation"
                r"|\bild\b",
    "bronchiect": r"bronchiectas",
    "mets": r"metasta|innumerable"
            r"|osseous lesion|lytic lesion"
            r"|sclerotic lesion",
    "lymphangitic": r"lymphangitic",
    "cirrhosis": r"cirrho|nodular (?:contour"
                 r"|liver)|hepatic steatosis",
    "ascites": r"ascites|peritoneal fluid",
    "pericard_eff": r"pericardial effusion"
                    r"|pericardial fluid",
    "aortic_ath": r"aortic (?:atheroscler"
                  r"|calcific)|atheroscler"
                  r"|calcified (?:aorta|plaque)",
    "aneurysm": r"aneurysm|dissect",
    "valve": r"valv\w+ (?:calcific|replace"
             r"|prosthe)|prosthetic valve"
             r"|annular calcific",
    "cachexia": r"cachexia|cachectic"
                r"|sarcopeni|muscle wasting",
    "obesity": r"obes|large body habitus"
               r"|body habitus",
    "renal": r"nephrostomy|atrophic kidney"
             r"|renal atroph|hydronephro",
    "pleural_thick": r"pleural thickening"
                     r"|pleural plaque",
    "vert_fx": r"compression (?:fracture|deform)"
               r"|vertebral fracture",
}


def norm(t):
    t = str(t)
    t = re.sub(r"<[^>]{1,20}>", " ", t)
    t = re.sub(r"_{2,}", " ", t)
    t = re.sub(r"^\s*IMPRESSION[S]?\s*:", " ",
               t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def get_imp(t):
    t = str(t)
    m = re.search(r"IMPRESSION[S]?\s*:", t,
                  flags=re.I)
    if not m:
        return ""
    s = t[m.end():]
    c = re.search(r"\n\s*(?:ADDENDUM"
                  r"|NOTIFICATION|WET READ"
                  r"|PRELIMINARY)", s, flags=re.I)
    if c:
        s = s[:c.start()]
    return s


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


def featurise(txt):
    low = txt.lower()
    cs = clauses(low)
    d = {}
    for k, p in BASER.items():
        d[k] = hit(cs, p)
    d["pe_neg"] = hit(cs, BASER["pe_pos"], True)
    d["txt_len"] = len(txt)
    d["n_sent"] = len(cs)
    for k, p in COMORB.items():
        d["cm_" + k] = hit(cs, p)
    return d


XC = (list(BASER.keys()) + ["pe_neg",
                            "txt_len", "n_sent"]
      + ["cm_" + k for k in COMORB])
print("features:", len(XC))


def make_model():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=10, cv=3, scoring="roc_auc",
            max_iter=5000, n_jobs=-1))])


def evalcv(X, y, grp):
    aucs = []
    ap = np.nan
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
            ap = average_precision_score(y, oof)
    return float(np.mean(aucs)), \
        float(np.std(aucs)), ap


# ---------- MIMIC side ----------
print("")
print("#" * 55)
print("MIMIC featurisation")
nt = pd.read_csv(NOTES)
for c in ["charttime", "admittime"]:
    nt[c] = pd.to_datetime(nt[c],
                           errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm", as_index=False).first()
nt["h"] = ((nt["charttime"] - nt["admittime"])
           .dt.total_seconds() / 3600.0)
nt = nt[nt["h"] <= POST_H]
print("presentation-window admissions:", len(nt))
nt["txt"] = nt["text"].apply(
    lambda t: norm(get_imp(t)))
nt = nt[nt["txt"].str.len() > 0]
print("with impression:", len(nt))

mrows = [featurise(t) for t in nt["txt"]]
mfx = pd.DataFrame(mrows)
mfx["hadm_id"] = nt["idx_hadm"].values
mfx.to_csv(MOUT, index=False)
print("saved", MOUT)

lab = pd.read_csv(LAB)
mdf = lab.merge(mfx, on="hadm_id")
print("merged:", len(mdf))

print("")
print("MIMIC-TRAINED REFERENCE (same features)")
MREF = {}
for oc in ["death_30d", "composite_30d",
           "cv_first", "death_30d_inhosp"]:
    dd = mdf
    if oc == "cv_first":
        dd = mdf[mdf["death_first"] == 0]
    y = dd[oc].values.astype(int)
    if y.sum() < 20:
        continue
    a, s, ap = evalcv(
        dd[XC].values.astype(float), y,
        dd["subject_id"].values)
    MREF[oc] = a
    print("  %-18s n=%d ev=%d AUC %.4f +/- %.4f"
          " AP %.4f"
          % (oc, len(y), int(y.sum()), a, s, ap))
pd.Series(MREF).to_csv(
    DATA + "/mimic_imp_ref_v2.csv")

# ---------- INSPECT side ----------
print("")
print("#" * 55)
print("INSPECT assembly")
imp = pd.read_csv(INS + "/impressions" + D,
                  sep="\t")
ilab = pd.read_csv(INS + "/labels" + D, sep="\t")
spl = pd.read_csv(INS + "/splits" + D, sep="\t")
mp = pd.read_csv(INS + "/study_mapping" + D,
                 sep="\t")
hl = pd.read_csv(DATA + "/inspect_labels_final.csv")
hl["pdate"] = pd.to_datetime(hl["pe_date"],
                             errors="coerce")

g = imp.merge(ilab, on="impression_id")
g = g.merge(spl, on="impression_id")
g = g.merge(mp[["impression_id",
                "procedure_DATETIME"]],
            on="impression_id")
g["pdt"] = pd.to_datetime(
    g["procedure_DATETIME"], errors="coerce")
g["txt"] = g["impressions"].apply(norm)
g = g[g["txt"].str.len() > 0]
print("studies:", len(g))

irows = [featurise(t) for t in g["txt"]]
ifx = pd.DataFrame(irows)
for c in XC:
    g[c] = ifx[c].values

# cell 1: PE-code persons, gap <= 30d
pe = g[g["person_id"].isin(set(hl["person_id"]))]
pe = pe.merge(
    hl[["person_id", "pdate", "death_30d",
        "composite_30d", "cv_first",
        "death_first"]],
    on="person_id", how="left")
pe["gap"] = (pe["pdt"] - pe["pdate"]).abs()
pe = pe[pe["gap"] <= pd.Timedelta(days=GAP_D)]
pe = pe.sort_values(["person_id", "gap"])
pe = pe.groupby("person_id",
                as_index=False).first()
print("CELL pe: %d persons (gap<=%dd)"
      % (len(pe), GAP_D))

# cell 2: all studies, first scan per person
al = g.sort_values(["person_id", "pdt"])
al = al.groupby("person_id",
                as_index=False).first()
print("CELL all: %d persons" % len(al))

print("")
print("PREVALENCE: INSPECT(pe) vs MIMIC")
for c in XC:
    if c in ("txt_len", "n_sent"):
        continue
    a = float(pe[c].mean())
    b = float(mdf[c].mean())
    fl = "  <--" if abs(a - b) > 0.10 else ""
    print("  %-16s INS %.3f  MIM %.3f%s"
          % (c, a, b, fl))

JOBS = [("pe", pe, "native_1m_mortality",
         "1_month_mortality"),
        ("pe", pe, "harm_death_30d",
         "death_30d"),
        ("pe", pe, "harm_composite_30d",
         "composite_30d"),
        ("pe", pe, "harm_cv_first",
         "cv_first"),
        ("all", al, "native_1m_mortality",
         "1_month_mortality")]

res = []
for cell, df, nm, col in JOBS:
    dd = df.copy()
    if col == "1_month_mortality":
        s = dd[col].astype(str).str.lower()
        dd = dd[s.isin(["true", "false"])]
        y = (dd[col].astype(str).str.lower()
             == "true").astype(int).values
    else:
        if col not in dd.columns:
            continue
        if col == "cv_first":
            dd = dd[dd["death_first"] == 0]
        y = pd.to_numeric(
            dd[col], errors="coerce"
        ).fillna(0).astype(int).values
    if y.sum() < 20:
        continue
    X = dd[XC].values.astype(float)
    grp = dd["person_id"].values
    a, s, ap = evalcv(X, y, grp)
    print("")
    print("=" * 55)
    print("[%s] %s n=%d ev=%d (%.4f)"
          % (cell, nm, len(y), int(y.sum()),
             y.mean()))
    print("  CV AUC %.4f +/- %.4f  AP %.4f"
          % (a, s, ap))
    md = make_model()
    md.fit(X, y)
    co = pd.Series(
        md.named_steps["lr"].coef_[0],
        index=XC).sort_values(ascending=False)
    print("  top +:", co.head(5).round(3).to_dict())
    print("  top -:", co.tail(3).round(3).to_dict())
    tag = cell + "_" + nm
    joblib.dump({"model": md, "cols": XC,
                 "target": col, "cell": cell},
                DATA + "/inspect_model_" + tag
                + ".joblib")
    res.append({"cell": cell, "model": nm,
                "target": col, "n": len(y),
                "ev": int(y.sum()), "auc": a,
                "sd": s, "ap": ap})

pd.DataFrame(res).to_csv(
    DATA + "/inspect_train_results.csv",
    index=False)
print("")
print(pd.DataFrame(res).round(4).to_string())
