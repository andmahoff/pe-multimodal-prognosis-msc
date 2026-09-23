import os
import sys
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import torchxrayvision as xrv
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

LABEL = sys.argv[1]
print("LABEL:", LABEL)

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
KEY = ["subject_id", "hadm_id"]

N_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 15
FREEZE_EPOCHS = 1
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 5
IMG_SIZE = 224
NUM_WORKERS = 4
SEED = 42
INNER_VAL = 0.15

OUT = os.path.join(
    FD, "cxr_harm_%s_preds.csv" % LABEL
)
CKPT = os.path.join(P2, "checkpoints_harm")
os.makedirs(CKPT, exist_ok=True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


class XRDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(
                size=IMG_SIZE,
                scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(
                p=0.5),
            transforms.RandomRotation(
                degrees=12),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)),
            transforms.RandomPerspective(
                distortion_scale=0.1, p=0.3),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(
            r["local_path"]).convert("L")
        im = im.resize((IMG_SIZE, IMG_SIZE))
        a = np.array(im, dtype=np.float32)
        a = xrv.datasets.normalize(a, 255)
        a = np.nan_to_num(a, nan=0.0)
        t = torch.from_numpy(a).unsqueeze(0)
        if self.is_train:
            t = self.aug(t)
        return {
            "image": t,
            "label": torch.tensor(
                r["_y"], dtype=torch.float32),
        }


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = xrv.models.DenseNet(
            weights="densenet121-res224-chex")
        nf = self.backbone.classifier.in_features
        self.classifier = nn.Sequential(
            nn.Linear(nf, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

    def freeze(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = True

    def forward(self, x):
        f = self.backbone.features(x)
        f = nn.functional.relu(f, inplace=True)
        o = nn.functional.adaptive_avg_pool2d(
            f, (1, 1))
        o = torch.flatten(o, 1)
        return torch.clamp(
            self.classifier(o), -10.0, 10.0)


def get_opt(m, frozen):
    if frozen:
        return optim.AdamW(
            m.classifier.parameters(),
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY)
    return optim.AdamW([
        {"params":
            m.backbone.features.parameters(),
         "lr": BACKBONE_LR},
        {"params": m.classifier.parameters(),
         "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)


def loader(df, train):
    return DataLoader(
        XRDataset(df, train),
        batch_size=BATCH_SIZE, shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=train)


def train_ep(m, dl, crit, opt, sc):
    m.train()
    tot = 0.0
    for b in dl:
        x = b["image"].to(DEVICE,
                          non_blocking=True)
        y = b["label"].to(
            DEVICE, non_blocking=True
        ).unsqueeze(1)
        opt.zero_grad()
        with autocast():
            o = m(x)
            loss = crit(o.float(), y.float())
        if torch.isnan(loss):
            continue
        sc.scale(loss).backward()
        sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            m.parameters(), max_norm=0.5)
        sc.step(opt)
        sc.update()
        tot += loss.item() * x.size(0)
    return tot / max(len(dl.dataset), 1)


def infer(m, dl):
    m.eval()
    ps, ys = [], []
    with torch.no_grad():
        for b in dl:
            o = m(b["image"].to(DEVICE))
            p = torch.sigmoid(o)
            ps.extend(
                p.cpu().numpy().flatten())
            ys.extend(b["label"].numpy())
    return (np.nan_to_num(np.array(ps),
                          nan=0.5),
            np.array(ys))


def sauc(y, p):
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return 0.5


print("Loading images + labels...")
full = pd.concat([
    pd.read_csv(P2 + "train.csv"),
    pd.read_csv(P2 + "val.csv"),
    pd.read_csv(P2 + "test.csv"),
], ignore_index=True)
lab = pd.read_csv(
    os.path.join(
        FD, "mimic_labels_harmonised.csv"))
full = full.merge(
    lab[KEY + [LABEL, "death_first"]],
    on=KEY, how="inner")
if LABEL == "cv_first":
    full = full[full["death_first"] == 0]
full["_y"] = full[LABEL].astype(int)
full = full.reset_index(drop=True)
print("  images:", len(full),
      " subjects:",
      full["subject_id"].nunique(),
      " pos images:", int(full["_y"].sum()))

sl = full.groupby(
    "subject_id")["_y"].max()
sid = sl.index.values
sy = sl.values
print("  subjects positive:", int(sy.sum()))

skf = StratifiedKFold(
    n_splits=N_FOLDS, shuffle=True,
    random_state=SEED)

parts = []
summ = []
for k, (a, b) in enumerate(
        skf.split(sid, sy), start=1):
    tr_df = full[full["subject_id"].isin(
        sid[a])]
    te_df = full[full["subject_id"].isin(
        sid[b])]
    subs = tr_df["subject_id"].unique()
    l2 = tr_df.groupby(
        "subject_id")["_y"].max().loc[
        subs].values
    itr, iva = train_test_split(
        subs, test_size=INNER_VAL,
        stratify=l2, random_state=SEED)
    d_itr = tr_df[tr_df["subject_id"].isin(itr)]
    d_iva = tr_df[tr_df["subject_id"].isin(iva)]

    print("\n=== Fold %d === train %d val %d "
          "test %d" % (k, len(d_itr),
                       len(d_iva), len(te_df)))
    l_itr = loader(d_itr, True)
    l_iva = loader(d_iva, False)
    l_out = loader(te_df, False)

    m = Net().to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    m.freeze()
    opt = get_opt(m, True)
    scaler = GradScaler()

    best, bep, bst, ni = 0.0, 0, None, 0
    for ep in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        if ep == FREEZE_EPOCHS + 1:
            m.unfreeze()
            opt = get_opt(m, False)
        tl = train_ep(m, l_itr, crit,
                      opt, scaler)
        pv, yv = infer(m, l_iva)
        va = sauc(yv, pv)
        print("  ep %02d [%.0fs] loss %.4f "
              "innerVAL %.4f"
              % (ep, time.time() - t0, tl, va),
              flush=True)
        if va > best:
            best, bep, ni = va, ep, 0
            bst = copy.deepcopy(m.state_dict())
        else:
            ni += 1
            if ni >= PATIENCE:
                print("  early stop", ep)
                break

    m.load_state_dict(bst)
    cp = os.path.join(
        CKPT, "cxr_%s_fold%d.pt" % (LABEL, k))
    torch.save({"state_dict": bst,
                "label": LABEL, "fold": k,
                "best_epoch": bep,
                "inner_val": best}, cp)
    print("  saved", cp, flush=True)
    po, yo = infer(m, l_out)
    print("  SELECTED ep %d | outer AUC %.4f"
          % (bep, sauc(yo, po)), flush=True)

    o = te_df[[
        "dicom_id", "subject_id", "hadm_id"
    ]].copy().reset_index(drop=True)
    o["_y"] = yo
    o["p_cxr"] = po
    o["fold"] = k
    parts.append(o)
    pd.concat(parts, ignore_index=True
              ).to_csv(OUT, index=False)
    summ.append({"fold": k, "best_epoch": bep,
                 "inner_val": best,
                 "outer_auc": sauc(yo, po)})

P = pd.concat(parts, ignore_index=True)
P.to_csv(OUT, index=False)
s = pd.DataFrame(summ)
print("\n" + "=" * 60)
print(s.to_string(index=False))
print("mean outer AUC %.4f +/- %.4f"
      % (s["outer_auc"].mean(),
         s["outer_auc"].std()))

print("\nAggregations:")
print("  image-level AUC %.4f"
      % sauc(P["_y"], P["p_cxr"]))
g = P.groupby(KEY).agg(
    y=("_y", "max"), p=("p_cxr", "mean"))
print("  admission (all imgs) AUC %.4f "
      "(n=%d)" % (sauc(g["y"], g["p"]),
                  len(g)))

tim = pd.read_csv(
    os.path.join(FD, "cxr_image_timing.csv")
)[["dicom_id", "rel"]]
# Known issue: rel is defined per admission, but this merge
# matches on dicom_id only, so a film taken during one
# admission also counts as "during" for any other admission
# it is linked to. fix_cxr_agg.py repeats the aggregation
# matching on dicom_id and hadm_id; its *_fixed.csv files
# are the ones reported.
D = P.merge(tim, on="dicom_id", how="left")
D = D[D["rel"] == "during"]
if len(D) > 0:
    h = D.groupby(KEY).agg(
        y=("_y", "max"), p=("p_cxr", "mean"))
    print("  admission (during only) AUC "
          "%.4f  AP %.4f (n=%d ev=%d)"
          % (sauc(h["y"], h["p"]),
             average_precision_score(
                 h["y"], h["p"]),
             len(h), int(h["y"].sum())))
    h.reset_index().rename(
        columns={"p": "p_cxr"}
    )[KEY + ["p_cxr"]].to_csv(
        os.path.join(
            FD, "p_cxr_harm_%s.csv" % LABEL),
        index=False)
print("\nSaved", OUT)
