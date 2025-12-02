import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATA_PATH = "/Users/darshan/Desktop/hushhhhhh/final_github_data.csv"   
OUT_DIR = Path(".")
RANDOM_STATE = 42
K = 2

df = pd.read_csv(DATA_PATH)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    raise ValueError("No numeric columns found in the input file.")

X = df[num_cols].copy()
n = X.shape[0]
rng = np.random.RandomState(RANDOM_STATE)
fit_size = min(2000, n)
fit_idx = rng.choice(n, size=fit_size, replace=False)
X_fit = X.values[fit_idx]

# prediction
kmeans = MiniBatchKMeans(n_clusters=K,random_state=RANDOM_STATE,n_init=20,batch_size=4096,max_iter=300)
kmeans.fit(X_fit)
labels = kmeans.predict(X.values)

# counts per cluster
print(f"Points per cluster (k={K}): " + ", ".join(f"{i}={c}" for i, c in enumerate(np.bincount(labels, minlength=K))))
df_out = df.copy()
df_out["cluster"] = labels

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "github_data_labeled.csv"
df_out.to_csv(out_path, index=False)

pca_size = min(5000, n)
pca_idx = rng.choice(n, size=pca_size, replace=False)
Xp = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X.values[pca_idx])
plt.figure()
plt.scatter(Xp[:, 0], Xp[:, 1], c=labels[pca_idx], s=10)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
