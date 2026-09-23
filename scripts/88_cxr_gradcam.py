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
FOLD = int(sys.argv[2]) if len(sys.argv) > 2 \
    else 1

P2 = "./phase2_mimic/"
FD = ("./"
      "fusion_workspace/data/")
CKPT = os.path.join(P2, "checkpoints_harm")
OUT = os.path.join(FD, "gradcam")
os.makedirs(OUT, exist_ok=True)

IMG_SIZE = 224
N_CASES = 12
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")
print("device:", DEVICE, "label:", LABEL,
      "fold:", FOLD)


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

    def forward(self, x):
        f = self.backbone.features(x)
        f = nn.functional.relu(f)
        o = nn.functional.adaptive_avg_pool2d(
            f, (1, 1))
        o = torch.flatten(o, 1)
        return torch.clamp(
            self.classifier(o), -10.0, 10.0)


def load_img(p):
    im = Image.open(p).convert("L")
    im = im.resize((IMG_SIZE, IMG_SIZE))
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
    cam = cam - cam.min()
    if float(cam.max()) > 0:
        cam = cam / cam.max()
    cam = torch.nn.functional.interpolate(
        cam[None, None],
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False)[0, 0]
    return (cam.cpu().numpy(),
            float(torch.sigmoid(out).item()))


cp = os.path.join(
    CKPT, "cxr_%s_fold%d.pt" % (LABEL, FOLD))
if not os.path.exists(cp):
    print("MISSING checkpoint:", cp)
    raise SystemExit(1)

ck = torch.load(cp, map_location=DEVICE)
model = Net().to(DEVICE)
model.load_state_dict(ck["state_dict"])
model.eval()
print("loaded fold %d (epoch %d, innerVAL"
      " %.4f)" % (ck["fold"], ck["best_epoch"],
                  ck["inner_val"]))

pr = pd.read_csv(
    FD + "cxr_harm_%s_preds.csv" % LABEL)
pr = pr[pr["fold"] == FOLD]
meta = pd.read_csv(
    P2 + "pe_cxr_final_dataset.csv")
cols = ["dicom_id", "local_path"]
if "ViewPosition" in meta.columns:
    cols.append("ViewPosition")
pr = pr.merge(meta[cols].drop_duplicates(
    "dicom_id"), on="dicom_id", how="left")
pr = pr[pr["local_path"].notna()]
print("fold images:", len(pr))

k = max(N_CASES // 3, 1)
sel = pd.concat([
    pr[pr["_y"] == 1].nlargest(
        k, "p_cxr").assign(grp="TP_high"),
    pr[pr["_y"] == 0].nlargest(
        k, "p_cxr").assign(grp="FP_high"),
    pr[pr["_y"] == 0].nsmallest(
        k, "p_cxr").assign(grp="TN_low")])

rows = []
for i, (_, r) in enumerate(sel.iterrows()):
    try:
        x = load_img(r["local_path"])
    except Exception as e:
        print("could not load image", r["dicom_id"], e)
        continue
    xb = x.unsqueeze(0).to(DEVICE)
    cam, p = gradcam(model, xb)

    img = x[0].numpy()
    rng = float(np.max(img) - np.min(img))
    img = (img - np.min(img)) / (rng + 1e-8)

    fig, ax = plt.subplots(
        1, 2, figsize=(8, 4))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title(
        "%s  p=%.3f  y=%d"
        % (r["grp"], p, int(r["_y"])),
        fontsize=9)
    ax[0].axis("off")
    ax[1].imshow(img, cmap="gray")
    ax[1].imshow(cam, cmap="jet", alpha=0.45)
    vp = r["ViewPosition"] if "ViewPosition" \
        in r.index else "?"
    ax[1].set_title("Grad-CAM  view=%s" % vp,
                    fontsize=9)
    ax[1].axis("off")
    fn = os.path.join(
        OUT, "%s_f%d_%02d_%s.png"
        % (LABEL, FOLD, i, r["grp"]))
    plt.tight_layout()
    plt.savefig(fn, dpi=110)
    plt.close()

    h, w = cam.shape
    tot = float(cam.sum()) + 1e-8
    rows.append({
        "dicom_id": r["dicom_id"],
        "grp": r["grp"], "y": int(r["_y"]),
        "p": p, "view": vp,
        "mass_upper": float(
            cam[:h // 3].sum()) / tot,
        "mass_mid": float(
            cam[h // 3:2 * h // 3].sum()) / tot,
        "mass_lower": float(
            cam[2 * h // 3:].sum()) / tot,
        "mass_border": float(
            cam.sum()
            - cam[20:-20, 20:-20].sum()) / tot,
        "file": os.path.basename(fn)})
    print("  %-8s p=%.3f y=%d -> %s"
          % (r["grp"], p, int(r["_y"]),
             os.path.basename(fn)))

if not rows:
    print("no cases produced")
    raise SystemExit(1)

t = pd.DataFrame(rows)
t.to_csv(OUT + "/cam_summary_%s_f%d.csv"
         % (LABEL, FOLD), index=False)
print("")
print("SALIENCY MASS BY REGION")
print(t.groupby("grp")[
    ["mass_upper", "mass_mid", "mass_lower",
     "mass_border"]].mean().round(3).to_string())
print("")
print("saved to", OUT)
