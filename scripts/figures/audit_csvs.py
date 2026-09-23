"""Lists every CSV in the data folders with its row
count and first columns, used to check file and column
names before the figure scripts were written.
"""
import os
import pandas as pd

ROOTS = [
    "./fusion_workspace/data",
    "./phase2_mimic",
]

for root in ROOTS:
    print("\n" + "=" * 62)
    print("ROOT:", root)
    print("=" * 62)
    if not os.path.isdir(root):
        print("  MISSING")
        continue
    names = sorted(os.listdir(root))
    for n in names:
        if not n.endswith(".csv"):
            continue
        p = os.path.join(root, n)
        try:
            df = pd.read_csv(p, nrows=3)
            nrow = sum(1 for _ in open(p)) - 1
            cols = list(df.columns)
            if len(cols) > 14:
                shown = cols[:14] + ["...+%d" % (len(cols) - 14)]
            else:
                shown = cols
            print("\n%-42s rows=%d" % (n, nrow))
            print("   " + ", ".join(str(c) for c in shown))
        except Exception as e:
            print("\n%-42s READ FAIL: %s" % (n, e))

