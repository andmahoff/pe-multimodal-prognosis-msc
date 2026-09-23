import sys
import platform

print("python:", sys.version.split()[0])
print("exe:", sys.executable)
print("node:", platform.node())
print("release:", platform.release())

import numpy as np
import pandas as pd
import sklearn
import scipy

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)
print("scipy:", scipy.__version__)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.RandomState(42)
X = rng.randn(400, 3)
y = (X[:, 0] + 0.4 * rng.randn(400) > 0)
y = y.astype(int)

m = LogisticRegression(max_iter=1000)
m.fit(X, y)
p = m.predict_proba(X)[:, 1]
print("smoke AUC:", round(roc_auc_score(y, p), 4))
print("ENV OK")

