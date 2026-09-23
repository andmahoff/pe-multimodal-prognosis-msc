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
OUT = DATA + "/ctpa_comorb_results.csv"

SEEDS = [42, 7, 13]
OUTS = ["cv_first", "composite_30d",
        "death_30d", "death_30d_inhosp"]

NEG = (r"\bno\b|\bnot\b|\bwithout\b|\bnone\b"
       r"|\bnor\b|\bnegative for\b|\babsent\b"
       r"|\bfree of\b|\bunremarkable\b"
       r"|\bresolved\b|\bruled out\b")

BASER = {
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
                r"|\bild\b|usual interstitial",
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

DEVICE = {
    "ett": r"endotracheal tube|\bett\b"
           r"|intubat",
    "trach": r"tracheostomy|trach tube",
    "cvc": r"central (?:venous|line)"
           r"|\bpicc\b|port-?a-?cath"
           r"|dialysis catheter"
           r"|swan-?ganz",
    "pacer": r"pacemaker|\bicd\b"
             r"|defibrillator|pacing lead",
    "sternotomy": r"sternotomy|\bcabg\b"
                  r"|sternal wire|median stern",
    "chesttube": r"chest tube|pigtail"
                 r"|thoracostomy",
    "ngtube": r"nasogastric|\bng tube\b"
              r"|feeding tube|\bogt\b",
    "ivcfilter": r"ivc filter|vena cava filter",
}


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


def both_text(t):
    t = str(t)
    f = sect(t, r"FINDINGS?\s*:",
             r"IMPRESSION[S]?\s*:")
    i = sect(t, r"IMPRESSION[S]?\s*:",
             r"\n\s*(?:ADDENDUM|NOTIFICATION"
             r"|WET READ|PRELIMINARY)")
    return (f + " " + i).strip()


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
    for k, p in DEVICE.items():
        d["dv_" + k] = hit(cs, p)
    return d


lab = pd.read_csv(LAB)
nt = pd.read_csv(NOTES)
nt["charttime"] = pd.to_datetime(
    nt["charttime"], errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm", as_index=False).first()
print("admissions:", len(nt))

nt["both"] = nt["text"].apply(both_text)
rows = [featurise(t) for t in nt["both"]]
fx = pd.DataFrame(rows)
fx["hadm_id"] = nt["idx_hadm"].values

BCOLS = (list(BASER.keys())
         + ["pe_neg", "txt_len", "n_sent"])
CCOLS = ["cm_" + k for k in COMORB]
DCOLS = ["dv_" + k for k in DEVICE]

print("")
print("comorbidity prevalence:")
print(fx[CCOLS].mean().round(3).to_string())
print("")
print("device prevalence:")
print(fx[DCOLS].mean().round(3).to_string())

d = lab.merge(fx, on="hadm_id", how="inner")
print("")
print("merged:", len(d))
d.to_csv(DATA + "/ctpa_comorb_features.csv",
         index=False)

SETS = {
    "base": BCOLS,
    "base_cm": BCOLS + CCOLS,
    "base_cm_dv": BCOLS + CCOLS + DCOLS,
}


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

    for nm, XC in SETS.items():
        XC = [c for c in XC if c in dd.columns]
        X = dd[XC].values.astype(float)
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
        print("  %-11s AUC %.4f +/- %.4f "
              "AP %.4f (%d feats)"
              % (nm, mu, sd, ap, X.shape[1]))
        res.append({"outcome": out, "set": nm,
                    "n": len(y),
                    "ev": int(y.sum()),
                    "auc": mu, "sd": sd,
                    "ap": ap,
                    "nfeat": X.shape[1]})
        po = pd.DataFrame({
            "subject_id": dd["subject_id"].values,
            "hadm_id": dd["hadm_id"].values,
            "p_ctpa": o42})
        po.to_csv(DATA + "/p_ctpa_" + nm + "_"
                  + out + ".csv", index=False)

    md = make_model()
    XC = [c for c in SETS["base_cm_dv"]
          if c in dd.columns]
    md.fit(dd[XC].values.astype(float), y)
    s = pd.Series(md.named_steps["lr"].coef_[0],
                  index=XC).sort_values(
        ascending=False)
    print("  top +:", s.head(6).round(3).to_dict())
    print("  top -:", s.tail(4).round(3).to_dict())

rdf = pd.DataFrame(res)
rdf.to_csv(OUT, index=False)
print("")
print(rdf.round(4).to_string())

print("")
print("BLOCK EFFECTS (AUC deltas)")
for out in rdf["outcome"].unique():
    s = rdf[rdf["outcome"] == out]

    def g(k):
        v = s[s["set"] == k]["auc"]
        return float(v.iloc[0]) if len(v) else np.nan

    a, b, c = g("base"), g("base_cm"), g("base_cm_dv")
    print("%-18s comorb %+.4f  device %+.4f"
          % (out, b - a, c - b))
print("saved", OUT)
