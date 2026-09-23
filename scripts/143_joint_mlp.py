import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
OUT = DATA + "/joint_mlp.csv"

SEED = 42
NBOOT = 2000
STEP = 0.05
VAR = "base_cm_dv"
POST_H = 24
JOBS = ["death_30d", "composite_30d"]
LATENT = 8
EPOCHS = 200
PATIENCE = 20
LR_ = 1e-3
WD = 1e-3
DROP = 0.4

TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]
CTBASE = ["pe_pos", "saddle", "central",
          "lobar", "segmental", "subseg",
          "bilateral", "rv_strain",
          "septal_bow", "reflux",
          "mpa_enlarge", "infarct",
          "effusion", "malignancy",
          "consolid", "atelect", "edema",
          "cardiomeg", "adenopathy",
          "pe_neg", "txt_len", "n_sent"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = torch.device("cpu")


def rk(x):
    return rankdata(x) / (len(x) + 1.0)


def norm_path(s):
    s = str(s).rstrip("/")
    return "/".join(s.split("/")[-4:])


def grid3():
    g = []
    n = int(round(1.0 / STEP))
    for i in range(n + 1):
        for j in range(n - i + 1):
            a, b = i * STEP, j * STEP
            g.append((a, b, max(1 - a - b, 0.0)))
    return g


G3 = grid3()


def wf(cols, w):
    s = np.zeros(len(cols[0]))
    for c, wi in zip(cols, w):
        s = s + wi * c
    return s


def boot(y, pa, pb, grp):
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(NBOOT):
        pk = rng.choice(us, len(us),
                        replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(roc_auc_score(y[ii], pa[ii])
                 - roc_auc_score(y[ii], pb[ii]))
    d = np.array(d)
    return (float(d.mean()),
            float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


class Branch(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        h = max(nout * 2, 8)
        self.net = nn.Sequential(
            nn.Linear(nin, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(DROP),
            nn.Linear(h, nout),
            nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Joint(nn.Module):
    def __init__(self, dims, latent=LATENT):
        super().__init__()
        self.b = nn.ModuleList(
            [Branch(d, latent) for d in dims])
        k = latent * len(dims)
        self.head = nn.Sequential(
            nn.Linear(k, latent),
            nn.BatchNorm1d(latent),
            nn.ReLU(),
            nn.Dropout(DROP),
            nn.Linear(latent, 1))

    def forward(self, xs):
        z = [bi(x) for bi, x in
             zip(self.b, xs)]
        return self.head(torch.cat(z, dim=1))


def fit_joint(Xtr, ytr, Xva, yva, dims):
    m = Joint(dims).to(DEV)
    pw = float((len(ytr) - ytr.sum())
               / max(ytr.sum(), 1))
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw]))
    opt = torch.optim.AdamW(
        m.parameters(), lr=LR_,
        weight_decay=WD)
    ttr = [torch.tensor(x,
                        dtype=torch.float32)
           for x in Xtr]
    tva = [torch.tensor(x,
                        dtype=torch.float32)
           for x in Xva]
    yt = torch.tensor(
        ytr, dtype=torch.float32).view(-1, 1)
    best, bstate, ni = -1.0, None, 0
    for ep in range(EPOCHS):
        m.train()
        opt.zero_grad()
        loss = crit(m(ttr), yt)
        loss.backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(
                m(tva)).numpy().ravel()
        try:
            v = roc_auc_score(yva, pv)
        except ValueError:
            v = 0.5
        if v > best:
            best, ni = v, 0
            bstate = {k: t.clone() for k, t
                      in m.state_dict().items()}
        else:
            ni += 1
            if ni >= PATIENCE:
                break
    if bstate is not None:
        m.load_state_dict(bstate)
    return m, best


lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
mm = pd.read_csv(
    DATA + "/mimic_expanded_feats.csv")
v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv"
).drop_duplicates("hadm_id")
vit = {"mean_temp": "temp", "mean_hr": "hr",
       "mean_sbp": "sbp", "mean_dbp": "dbp",
       "mean_rr": "rr"}
keep = ["hadm_id"] + [c for c in vit
                      if c in v7.columns] \
    + [c for c in FLAGS if c in v7.columns]
mm = mm.merge(v7[keep].rename(columns=vit),
              on="hadm_id", how="left")
age = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v6.csv",
    usecols=["hadm_id", "age_at_admit"])
mm = mm.merge(
    age.dropna().drop_duplicates("hadm_id")
    .rename(columns={"age_at_admit": "age"}),
    on="hadm_id", how="left")

eg = pd.read_csv(P2 + "/bench_feats_logit.csv")
coh = pd.read_csv(
    P2 + "/mimic_pe_mace_cohort.csv",
    usecols=["subject_id", "hadm_id",
             "ecg_path"])
eg["k"] = eg["ecg_path"].apply(norm_path)
eg = eg.drop_duplicates("k")
coh["k"] = coh["ecg_path"].apply(norm_path)
ecgm = coh.merge(eg.drop(columns=["ecg_path"]),
                 on="k", how="inner")
LG = [c for c in ecgm.columns
      if c not in ("subject_id", "hadm_id",
                   "ecg_path", "k")]
ecgm = ecgm.groupby("hadm_id",
                    as_index=False)[LG].mean()

fx = pd.read_csv(
    DATA + "/ctpa_comorb_features.csv")
nt = pd.read_csv(
    DATA + "/ctpa_notes_index.csv")
for c in ["charttime", "admittime"]:
    nt[c] = pd.to_datetime(nt[c],
                           errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm",
                as_index=False).first()
nt["h"] = ((nt["charttime"] - nt["admittime"])
           .dt.total_seconds() / 3600.0)
nt = nt[nt["h"] <= POST_H]
fx = fx[fx["hadm_id"].isin(nt["idx_hadm"])]
CT = [c for c in fx.columns
      if c in CTBASE or c.startswith("cm_")
      or c.startswith("dv_")]

EC = [c for c in
      (["mi_" + a if "mi_" + a in mm.columns
        else a for a in TA] + FLAGS + ["age"])
      if c in mm.columns]
print("EHR:", len(EC), " ECG:", len(LG),
      " CTPA:", len(CT))

rows = []
for ycol in JOBS:
    pe = pd.read_csv(
        DATA + "/p_ehr_harm_" + ycol
        + ".csv")[["hadm_id", "p_ehr"]]
    pg = pd.read_csv(
        DATA + "/p_ecg_harm_" + ycol
        + ".csv")[["hadm_id", "p_ecg"]]
    pc = pd.read_csv(
        DATA + "/p_ctpa_pres_" + VAR + "_"
        + ycol + ".csv")[["hadm_id",
                          "p_ctpa"]]
    d = lab.merge(mm, on=["subject_id",
                          "hadm_id"])
    d = d.merge(ecgm, on="hadm_id")
    d = d.merge(fx[["hadm_id"] + CT],
                on="hadm_id")
    d = d.merge(pe, on="hadm_id")
    d = d.merge(pg, on="hadm_id")
    d = d.merge(pc, on="hadm_id")
    d = d.reset_index(drop=True)

    y = d[ycol].values.astype(int)
    grp = d["subject_id"].values
    blocks = [d[EC].values.astype(float),
              d[LG].values.astype(float),
              d[CT].values.astype(float)]
    dims = [b.shape[1] for b in blocks]

    print("")
    print("#" * 58)
    print("%s  n=%d ev=%d  dims %s"
          % (ycol, len(y), int(y.sum()),
             dims))

    a = rk(d["p_ehr"].values)
    b = rk(d["p_ecg"].values)
    c = rk(d["p_ctpa"].values)

    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    folds = list(cv.split(
        np.zeros((len(y), 1)), y, grp))

    ref = np.zeros(len(y))
    for tr, te in folds:
        best, bw = None, None
        for w in G3:
            v = roc_auc_score(
                y[tr], wf([a[tr], b[tr],
                           c[tr]], w))
            if best is None or v > best:
                best, bw = v, w
        ref[te] = wf([a[te], b[te], c[te]], bw)
    a_ref = roc_auc_score(y, ref)

    oof = np.zeros(len(y))
    vals = []
    for fi, (tr, te) in enumerate(folds):
        sub = np.unique(grp[tr])
        ymax = pd.Series(y[tr]).groupby(
            pd.Series(grp[tr])).max()
        itr, iva = train_test_split(
            sub, test_size=0.15,
            stratify=ymax.loc[sub].values,
            random_state=SEED)
        mtr = np.isin(grp, itr)
        mva = np.isin(grp, iva)

        Xtr, Xva, Xte = [], [], []
        for B in blocks:
            im = SimpleImputer(
                strategy="median").fit(B[mtr])
            sc = StandardScaler().fit(
                im.transform(B[mtr]))
            Xtr.append(sc.transform(
                im.transform(B[mtr])))
            Xva.append(sc.transform(
                im.transform(B[mva])))
            Xte.append(sc.transform(
                im.transform(B[te])))

        m, bv = fit_joint(
            Xtr, y[mtr], Xva, y[mva], dims)
        vals.append(bv)
        m.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(m(
                [torch.tensor(
                    x, dtype=torch.float32)
                 for x in Xte])).numpy().ravel()
        print("  fold %d innerVAL %.4f"
              % (fi + 1, bv))

    a_j = roc_auc_score(y, oof)
    ap_j = average_precision_score(y, oof)
    t = boot(y, oof, ref, grp)
    print("")
    print("  WMEAN3 (late)   %.4f  AP %.4f"
          % (a_ref,
             average_precision_score(y, ref)))
    print("  Joint MLP       %.4f  AP %.4f"
          % (a_j, ap_j))
    print("  joint vs late  %+.4f"
          " [%+.4f, %+.4f]" % t)
    rows.append({
        "outcome": ycol, "late": a_ref,
        "joint": a_j, "ap_joint": ap_j,
        "diff": t[0], "lo": t[1],
        "hi": t[2],
        "mean_innerval": float(
            np.mean(vals))})

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string())
print("saved", OUT)
