import re
import os
import glob
import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
INS = BASE + "/inspect_files"
DATA = BASE + "/fusion_workspace/data"
D = "_20250611.tsv"
FEAT = DATA + "/mimic_imp_features_v2.csv"
LAB = DATA + "/mimic_labels_harmonised.csv"
REFF = DATA + "/mimic_imp_ref_v2.csv"
OUT = DATA + "/inspect_transfer_results.csv"

NBOOT = 2000
SEED = 42
GAP_D = 30
DROP4 = ["pe_pos", "segmental", "subseg",
         "pe_neg"]

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


def rk(x):
    return rankdata(x) / (len(x) + 1.0)


def boot(y, p, grp):
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(SEED)
    b = []
    for _ in range(NBOOT):
        pk = rng.choice(us, len(us),
                        replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        b.append(roc_auc_score(y[ii], p[ii]))
    b = np.array(b)
    return (float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)))


ref = pd.read_csv(REFF, index_col=0)
REF = ref.iloc[:, 0].to_dict()
lab = pd.read_csv(LAB)
fx = pd.read_csv(FEAT)
print("mimic rows:", len(fx))

TGT = {"1_month_mortality": "death_30d",
       "death_30d": "death_30d",
       "composite_30d": "composite_30d",
       "cv_first": "cv_first"}

rows = []
store = {}


def record(tag, cell, ycol, y, p, grp, n):
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    lo, hi = boot(y, p, grp)
    r = REF.get(ycol, np.nan)
    print("  %-34s AUC %.4f [%.4f,%.4f] "
          "AP %.4f  cost %+.4f"
          % (tag, auc, lo, hi, ap, auc - r))
    rows.append({"variant": tag, "cell": cell,
                 "outcome": ycol, "n": n,
                 "ev": int(y.sum()),
                 "auc": auc, "lo": lo, "hi": hi,
                 "ap": ap, "ref": r,
                 "cost": auc - r})


print("")
print("#" * 58)
print("VARIANTS A (pure) and B (target scaler)")
for mp in sorted(glob.glob(
        DATA + "/inspect_model_*.joblib")):
    obj = joblib.load(mp)
    md = obj["model"]
    XC = obj["cols"]
    ycol = TGT.get(obj["target"])
    cell = obj.get("cell", "?")
    tag = os.path.basename(mp)[14:-7]
    if ycol is None:
        continue
    if any(c not in fx.columns for c in XC):
        continue

    d = lab.merge(fx[["hadm_id"] + XC],
                  on="hadm_id")
    if ycol == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)
    y = d[ycol].values.astype(int)
    X = d[XC].values.astype(float)
    grp = d["subject_id"].values

    print("")
    print("%s -> %s (n=%d ev=%d)"
          % (tag, ycol, len(y), int(y.sum())))

    pA = md.predict_proba(X)[:, 1]
    record("A_pure|" + tag, cell, ycol, y, pA,
           grp, len(y))
    store[("A", tag, ycol)] = (d, pA)

    im = md.named_steps["im"]
    lr = md.named_steps["lr"]
    Xi = im.transform(X)
    sc2 = StandardScaler().fit(Xi)
    pB = lr.predict_proba(
        sc2.transform(Xi))[:, 1]
    record("B_tgtscale|" + tag, cell, ycol, y,
           pB, grp, len(y))
    store[("B", tag, ycol)] = (d, pB)

    po = pd.DataFrame({
        "subject_id": d["subject_id"].values,
        "hadm_id": d["hadm_id"].values,
        "p_zs_A": pA, "p_zs_B": pB})
    po.to_csv(DATA + "/p_ctpa_zs_" + tag
              + ".csv", index=False)

print("")
print("#" * 58)
print("VARIANT C: ensemble pe + all")
for v in ["A", "B"]:
    k1 = (v, "pe_native_1m_mortality",
          "death_30d")
    k2 = (v, "all_native_1m_mortality",
          "death_30d")
    if k1 not in store or k2 not in store:
        continue
    d1, p1 = store[k1]
    d2, p2 = store[k2]
    m = d1[["hadm_id"]].copy()
    m["p1"] = rk(p1)
    m2 = d2[["hadm_id"]].copy()
    m2["p2"] = rk(p2)
    m = m.merge(m2, on="hadm_id")
    m = m.merge(lab[["hadm_id", "subject_id",
                     "death_30d"]],
                on="hadm_id")
    y = m["death_30d"].values.astype(int)
    p = (m["p1"].values + m["p2"].values) / 2.0
    record("C_ens_" + v, "pe+all", "death_30d",
           y, p, m["subject_id"].values, len(y))

print("")
print("#" * 58)
print("VARIANT D: prevalence-stable subset")
imp = pd.read_csv(INS + "/impressions" + D,
                  sep="\t")
il = pd.read_csv(INS + "/labels" + D, sep="\t")
spl = pd.read_csv(INS + "/splits" + D, sep="\t")
mpp = pd.read_csv(INS + "/study_mapping" + D,
                  sep="\t")
hl = pd.read_csv(
    DATA + "/inspect_labels_final.csv")
hl["pdate"] = pd.to_datetime(hl["pe_date"],
                             errors="coerce")

g = imp.merge(il, on="impression_id")
g = g.merge(spl, on="impression_id")
g = g.merge(mpp[["impression_id",
                 "procedure_DATETIME"]],
            on="impression_id")
g["pdt"] = pd.to_datetime(
    g["procedure_DATETIME"], errors="coerce")
g["txt"] = g["impressions"].apply(norm)
g = g[g["txt"].str.len() > 0]
gf = pd.DataFrame(
    [featurise(t) for t in g["txt"]])
ALLC = list(gf.columns)
for c in ALLC:
    g[c] = gf[c].values

pe = g[g["person_id"].isin(set(hl["person_id"]))]
pe = pe.merge(hl[["person_id", "pdate"]],
              on="person_id", how="left")
pe["gap"] = (pe["pdt"] - pe["pdate"]).abs()
pe = pe[pe["gap"] <= pd.Timedelta(days=GAP_D)]
pe = pe.sort_values(["person_id", "gap"])
pe = pe.groupby("person_id",
                as_index=False).first()
print("INSPECT pe cell:", len(pe))

XR = [c for c in ALLC if c not in DROP4]
print("reduced features:", len(XR))

s = pe["1_month_mortality"].astype(
    str).str.lower()
tr = pe[s.isin(["true", "false"])]
yt = (tr["1_month_mortality"].astype(str)
      .str.lower() == "true").astype(int).values
md = Pipeline([
    ("im", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("lr", LogisticRegressionCV(
        Cs=10, cv=3, scoring="roc_auc",
        max_iter=5000, n_jobs=-1))])
md.fit(tr[XR].values.astype(float), yt)

d = lab.merge(fx[["hadm_id"] + XR],
              on="hadm_id").reset_index(drop=True)
y = d["death_30d"].values.astype(int)
X = d[XR].values.astype(float)
pD = md.predict_proba(X)[:, 1]
record("D_stable|pe_native", "pe", "death_30d",
       y, pD, d["subject_id"].values, len(y))

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print("")
print(res.round(4).to_string())
print("saved", OUT)
