import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchxrayvision as xrv
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = sys.argv[1] if len(sys.argv) > 1 \
    else "death_30d"

P2 = "./phase2_mimic/"
FD = ("./"
      "fusion_workspace/data/")
CKPT = os.path.join(P2, "checkpoints_harm")
OUT = os.path.join(FD, "gradcam_fig")
os.makedirs(OUT, exist_ok=True)

IMG = 224
PER_GRP = 30
BORDER = 20
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")

frac = ((IMG * IMG - (IMG - 2 * BORDER) ** 2)
        / float(IMG * IMG))
print("device:", DEVICE, "label:", LABEL)
print("uniform border baseline: %.3f" % frac)


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
            nn.Linear(256, 1))


def load_img(p):
    im = Image.open(p).convert("L")
    im = im.resize((IMG, IMG))
    a = np.array(im, dtype=np.float32)
    a = xrv.datasets.normalize(a, 255)
    a = np.nan_to_num(a, nan=0.0)
    return torch.from_numpy(a).unsqueeze(0)


def gradcam(model, x):
    torch.set_grad_enabled(True)
    model.zero_grad()
    A = model.backbone.features(x)
    A.retain_grad()
    f = nn.functional.relu(A)
    o = nn.functional.adaptive_avg_pool2d(
        f, (1, 1))
    o = torch.flatten(o, 1)
    out = torch.clamp(
        model.classifier(o), -10.0, 10.0)
    out.sum().backward()
    Ad = A.detach()[0]
    Gd = A.grad.detach()[0]
    w = Gd.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None]
                      * Ad).sum(0))
    s = float(cam.sum())
    if s <= 1e-6:
        return None, float(
            torch.sigmoid(out).item())
    cam = cam / cam.max()
    cam = torch.nn.functional.interpolate(
        cam[None, None], size=(IMG, IMG),
        mode="bilinear",
        align_corners=False)[0, 0]
    return (cam.cpu().numpy(),
            float(torch.sigmoid(out).item()))


meta = pd.read_csv(
    P2 + "pe_cxr_final_dataset.csv")
mc = ["dicom_id", "local_path"]
for c in ["ViewPosition",
          "PerformedProcedureStepDescription"]:
    if c in meta.columns:
        mc.append(c)
meta = meta[mc].drop_duplicates("dicom_id")
print("meta cols:", mc)

pr = pd.read_csv(
    FD + "cxr_harm_%s_preds.csv" % LABEL)
pr = pr.merge(meta, on="dicom_id", how="left")
pr = pr[pr["local_path"].notna()]
print("all fold images:", len(pr))

acc = {}
cnt = {}
degen = {}
rows = []
exemplar = {}

for fold in sorted(pr["fold"].unique()):
    cp = os.path.join(
        CKPT, "cxr_%s_fold%d.pt"
        % (LABEL, fold))
    if not os.path.exists(cp):
        continue
    ck = torch.load(cp, map_location=DEVICE)
    model = Net().to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    pf = pr[pr["fold"] == fold]
    pf = pf.sort_values(
        "p_cxr", ascending=False)
    pf = pf.drop_duplicates("subject_id")

    grps = {
        "high_event": pf[pf["_y"] == 1].head(
            PER_GRP),
        "high_nonevent": pf[
            pf["_y"] == 0].head(PER_GRP),
        "low_risk": pf[pf["_y"] == 0].tail(
            PER_GRP)}

    for g, sub in grps.items():
        for _, r in sub.iterrows():
            try:
                x = load_img(r["local_path"])
            except Exception:
                continue
            cam, p = gradcam(
                model, x.unsqueeze(0).to(DEVICE))
            if cam is None:
                degen[g] = degen.get(g, 0) + 1
                continue
            acc[g] = acc.get(
                g, np.zeros((IMG, IMG))) + cam
            cnt[g] = cnt.get(g, 0) + 1
            tot = cam.sum()
            b = (cam.sum()
                 - cam[BORDER:-BORDER,
                       BORDER:-BORDER].sum())
            h = IMG
            rows.append({
                "fold": fold, "grp": g,
                "p": p, "y": int(r["_y"]),
                "upper": cam[:h // 3].sum() / tot,
                "mid": cam[h // 3:2 * h
                           // 3].sum() / tot,
                "lower": cam[2 * h
                             // 3:].sum() / tot,
                "border": b / tot})
            if g not in exemplar and \
                    r["_y"] == int(
                        g == "high_event"):
                exemplar[g] = (
                    x[0].numpy(), cam, p,
                    int(r["_y"]))
    print("fold %d done (valid %d)"
          % (fold, sum(cnt.values())))

t = pd.DataFrame(rows)
t.to_csv(OUT + "/cam_cases_%s.csv" % LABEL,
         index=False)

print("")
print("VALID / DEGENERATE CAMs")
for g in acc:
    print("  %-14s valid %3d  degenerate %3d"
          % (g, cnt.get(g, 0),
             degen.get(g, 0)))

print("")
print("REGIONAL SALIENCY MASS"
      " (uniform border = %.3f)" % frac)
s = t.groupby("grp")[
    ["upper", "mid", "lower", "border"]]
m = s.mean().round(3)
sd = s.std().round(3)
n = s.size()
for g in m.index:
    print("  %-14s n=%3d  upper %.3f  mid"
          " %.3f  lower %.3f  border %.3f"
          " (+/- %.3f)"
          % (g, n[g], m.loc[g, "upper"],
             m.loc[g, "mid"],
             m.loc[g, "lower"],
             m.loc[g, "border"],
             sd.loc[g, "border"]))
m.to_csv(OUT + "/cam_regions_%s.csv" % LABEL)

order = ["high_event", "high_nonevent",
         "low_risk"]
order = [g for g in order if g in acc]
fig, ax = plt.subplots(
    2, len(order),
    figsize=(4 * len(order), 8))
if len(order) == 1:
    ax = ax.reshape(2, 1)

for j, g in enumerate(order):
    mean_cam = acc[g] / max(cnt[g], 1)
    mean_cam = mean_cam / (
        mean_cam.max() + 1e-8)
    im = ax[0, j].imshow(mean_cam, cmap="jet")
    ax[0, j].set_title(
        "mean Grad-CAM\n%s  (n=%d)"
        % (g, cnt[g]), fontsize=10)
    ax[0, j].axis("off")
    plt.colorbar(im, ax=ax[0, j],
                 fraction=0.046)
    if g in exemplar:
        img, cam, p, yy = exemplar[g]
        rr = float(np.max(img) - np.min(img))
        img = (img - np.min(img)) / (rr + 1e-8)
        ax[1, j].imshow(img, cmap="gray")
        ax[1, j].imshow(cam, cmap="jet",
                        alpha=0.45)
        ax[1, j].set_title(
            "example  p=%.3f  y=%d"
            % (p, yy), fontsize=10)
    ax[1, j].axis("off")

plt.suptitle(
    "CXR Grad-CAM, %s (all folds pooled)"
    % LABEL, fontsize=12)
plt.tight_layout()
fn = OUT + "/gradcam_%s.png" % LABEL
plt.savefig(fn, dpi=150)
plt.close()
print("")
print("figure:", fn)

np.save(OUT + "/meancam_%s.npy" % LABEL,
        np.stack([acc[g] / max(cnt[g], 1)
                  for g in order]))
print("saved to", OUT)
