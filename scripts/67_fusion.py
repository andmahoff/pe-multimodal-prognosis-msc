import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
KEY = ["subject_id", "hadm_id"]
SEED = 42
NBOOT = 1000

EHR_VARIANTS = ["vrex", "dann_vrex"]


def rank01(x):
    return pd.Series(x).rank(pct=True).values


print("Building fusion table...")
tm = pd.read_csv(
    os.path.join(FD, "cxr_timing_test.csv")
)
coh = tm[tm["n_dur"] > 0][
    KEY + ["y", "n_dur", "n_all"]
].copy()
coh = coh.rename(columns={"y": "label"})
print("  cohort admissions:", len(coh))

cx = pd.read_csv(
    os.path.join(FD, "cxr_oof_predictions.csv")
)
meta = pd.concat([
    pd.read_csv(P2 + "train.csv"),
    pd.read_csv(P2 + "val.csv"),
    pd.read_csv(P2 + "test.csv"),
], ignore_index=True)
meta = meta[[
    "dicom_id", "StudyDate",
    "PerformedProcedureStepDescription",
]].drop_duplicates("dicom_id")
cx = cx.merge(meta, on="dicom_id", how="left")
cx["port"] = (
    cx["PerformedProcedureStepDescription"]
    .fillna("").str.upper()
    .str.contains("PORT").astype(int)
)

cxa = cx.groupby(KEY).agg(
    p_cxr_all=("p_cxr", "mean"),
    frac_port=("port", "mean"),
).reset_index()
coh = coh.merge(cxa, on=KEY, how="left")

print("  p_cxr is the mean over all images;"
      " index-stay filtering is applied in"
      " 68_ro4_evaluation.py")

ecg = pd.read_csv(
    os.path.join(P2, "logit_final_pecg.csv")
)
ecga = ecg.groupby(KEY).agg(
    p_ecg=("p_ecg", "mean"),
).reset_index()
coh = coh.merge(ecga, on=KEY, how="left")
print("  after ECG merge, nulls:",
      coh["p_ecg"].isna().sum())


def boot_diff(y, pa, pb, groups, n=NBOOT):
    rng = np.random.RandomState(SEED)
    g = np.array(groups)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    out = []
    for _ in range(n):
        s = rng.choice(uq, len(uq),
                       replace=True)
        take = np.concatenate([idx[u]
                               for u in s])
        yy = y[take]
        if len(np.unique(yy)) < 2:
            continue
        out.append(
            roc_auc_score(yy, pa[take])
            - roc_auc_score(yy, pb[take])
        )
    out = np.array(out)
    return out.mean(), np.percentile(
        out, [2.5, 97.5]
    )


results = []

for ev in EHR_VARIANTS:
    print("\n" + "=" * 60)
    print("EHR VARIANT:", ev)
    print("=" * 60)

    eh = pd.read_csv(os.path.join(
        FD, "p_ehr_ensemble_v7_%s.csv" % ev
    ))[KEY + ["p_ehr_mean"]]
    d = coh.merge(eh, on=KEY, how="left")
    d = d.dropna(subset=[
        "p_ehr_mean", "p_ecg", "p_cxr_all"
    ]).reset_index(drop=True)
    print("  usable rows:", len(d),
          "events:", int(d["label"].sum()))

    y = d["label"].values
    grp = d["subject_id"].values

    d["r_ehr"] = rank01(d["p_ehr_mean"])
    d["r_ecg"] = rank01(d["p_ecg"])
    d["r_cxr"] = rank01(d["p_cxr_all"])
    d["r_port"] = rank01(d["frac_port"])
    d["r_ndur"] = rank01(d["n_dur"])

    base = {
        "EHR": d["r_ehr"].values,
        "ECG": d["r_ecg"].values,
        "CXR": d["r_cxr"].values,
        "META(port+n)": (
            d["r_port"].values
            + d["r_ndur"].values
        ) / 2,
    }

    d["fuse_mean3"] = (
        d["r_ehr"] + d["r_ecg"] + d["r_cxr"]
    ) / 3
    d["fuse_mean2"] = (
        d["r_ehr"] + d["r_ecg"]
    ) / 2

    skf = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED
    )
    stacks = {
        "STACK3": ["r_ehr", "r_ecg", "r_cxr"],
        "STACK3+META": [
            "r_ehr", "r_ecg", "r_cxr",
            "r_port", "r_ndur"
        ],
        "STACK2(EHR+ECG)": ["r_ehr", "r_ecg"],
    }
    for name, cols in stacks.items():
        oof = np.zeros(len(d))
        for tr, te in skf.split(
            d[cols], y, groups=grp
        ):
            lr = LogisticRegression(
                max_iter=2000, C=1.0
            )
            lr.fit(d[cols].values[tr], y[tr])
            oof[te] = lr.predict_proba(
                d[cols].values[te]
            )[:, 1]
        d[name] = oof

    allm = dict(base)
    allm["MEAN3"] = d["fuse_mean3"].values
    allm["MEAN2(EHR+ECG)"] = (
        d["fuse_mean2"].values
    )
    for name in stacks:
        allm[name] = d[name].values

    print("\n  %-18s %-8s %-8s"
          % ("model", "AUC", "AUPRC"))
    for k, v in allm.items():
        results.append({
            "ehr_variant": ev, "model": k,
            "auc": roc_auc_score(y, v),
            "ap": average_precision_score(y, v),
            "n": len(d),
            "events": int(y.sum()),
        })
        print("  %-18s %.4f   %.4f"
              % (k, roc_auc_score(y, v),
                 average_precision_score(y, v)))

    best_uni = max(
        ["EHR", "ECG", "CXR"],
        key=lambda k: roc_auc_score(y, allm[k])
    )
    print("\n  best unimodal:", best_uni)
    print("\n  paired bootstrap vs %s "
          "(1000 subject-level resamples):"
          % best_uni)
    for k in ["MEAN3", "STACK3",
              "STACK3+META", "MEAN2(EHR+ECG)",
              "STACK2(EHR+ECG)"]:
        m, ci = boot_diff(
            y, allm[k], allm[best_uni], grp
        )
        sig = "*" if (ci[0] > 0) else " "
        print("   %-18s %+0.4f "
              "[%+0.4f, %+0.4f] %s"
              % (k, m, ci[0], ci[1], sig))

    print("\n  CXR increment "
          "(STACK3 vs STACK2):")
    m, ci = boot_diff(
        y, allm["STACK3"],
        allm["STACK2(EHR+ECG)"], grp
    )
    print("   %+0.4f [%+0.4f, %+0.4f]"
          % (m, ci[0], ci[1]))

    print("\n  CXR vs META alone:")
    m, ci = boot_diff(
        y, allm["CXR"],
        allm["META(port+n)"], grp
    )
    print("   %+0.4f [%+0.4f, %+0.4f]"
          % (m, ci[0], ci[1]))

    print("\n  rank correlations:")
    cc = d[["r_ehr", "r_ecg", "r_cxr"]].corr(
        method="spearman"
    )
    print(cc.to_string())

r = pd.DataFrame(results)
out = os.path.join(FD, "fusion_results.csv")
r.to_csv(out, index=False)
print("\nSaved", out)
