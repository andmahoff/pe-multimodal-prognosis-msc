import csv
import os
import re

SD = ("./"
      "fusion_workspace/scripts/")
FD = ("./"
      "fusion_workspace/data/")
OUT = FD + "ctpa_patterns.csv"

SCRIPT = "100_ctpa_comorb.py"

# variable name -> block label for the table
BLOCKS = [
    ("BASER",  "PE descriptors and incidental findings"),
    ("COMORB", "Comorbidity mentions"),
    ("DEVICE", "Support-device mentions"),
]

src = open(os.path.join(SD, SCRIPT)).read().split("\n")


def grab(name):
    """Return the dict literal for `name` as text."""
    for i, l in enumerate(src):
        if re.match(r"\s*%s\s*=\s*\{\s*$" % name, l):
            depth = 1
            j = i + 1
            out = [l.strip()]
            while j < len(src) and depth > 0:
                depth += (src[j].count("{")
                          - src[j].count("}"))
                out.append(src[j])
                j += 1
            return "\n".join(out)
    return None


rows = []
for var, label in BLOCKS:
    block = grab(var)
    if block is None:
        print("NOT FOUND:", var)
        continue
    ns = {}
    exec(block, {"re": re}, ns)
    d = ns[var]
    print("%-8s %2d patterns" % (var, len(d)))
    for feat, pat in d.items():
        if isinstance(pat, (list, tuple)):
            pat = " | ".join(str(p) for p in pat)
        rows.append({"block": label,
                     "feature": feat,
                     "pattern": str(pat)})

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(
        f, ["block", "feature", "pattern"])
    w.writeheader()
    w.writerows(rows)

print("\nwrote:", OUT, "|", len(rows), "patterns\n")
for r in rows:
    print("  %-24s %-18s %s"
          % (r["block"][:22], r["feature"],
             r["pattern"][:52]))

