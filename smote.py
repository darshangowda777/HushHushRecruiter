import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTENC
from pathlib import Path

IN_PATH = "/Users/darshan/Desktop/hushhhhhh/github_data_labeled.csv"
OUT_PATH = "/Users/darshan/Desktop/hushhhhhh/github_data_labeled_smoted.csv"

data = pd.read_csv(IN_PATH)

if "cluster" not in data.columns:
    raise KeyError("Expected a 'cluster' column as the target label.")

y = data["cluster"]
X = data.drop(columns=["cluster"])

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for must_have in ["owner", "email"]:
    if must_have in X.columns and must_have not in cat_cols:
        cat_cols.append(must_have)

if len(cat_cols) == 0:
    raise ValueError(
        "No categorical columns detected. If you truly have none, use SMOTE instead of SMOTENC. "
        "Otherwise ensure 'owner'/'email' are present or dtype=object/category."
    )

categorical_features = [X.columns.get_loc(c) for c in cat_cols]

# BEFORE distribution
count_class = y.value_counts().sort_index()
plt.bar(count_class.index, count_class.values)
plt.xlabel('Class'); plt.ylabel('Count'); plt.title('Class Distribution (before SMOTE)')
plt.xticks(count_class.index, [f'Class {i}' for i in count_class.index])
plt.tight_layout(); plt.show()

# Choose k_neighbors
n_minority = count_class.min()
k_neighbors = max(1, min(5, n_minority - 1))
if k_neighbors < 1:
    raise ValueError(
        "The smallest class has only 1 sample. Need at least 2 to run SMOTENC. "
        "Consider using RandomOverSampler or collect more data."
    )

smote = SMOTENC(categorical_features=categorical_features,sampling_strategy="minority",k_neighbors=k_neighbors,random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# AFTER distribution 
count_after = y_res.value_counts().sort_index()
plt.bar(count_after.index, count_after.values)
plt.xlabel('Class'); plt.ylabel('Count'); plt.title('Class Distribution (after SMOTE)')
plt.xticks(count_after.index, [f'Class {i}' for i in count_after.index])
plt.tight_layout(); plt.show()

resampled = pd.concat([pd.DataFrame(X_res, columns=X.columns), y_res.rename("cluster")], axis=1)
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
resampled.to_csv(OUT_PATH, index=False)
print(f"Saved balanced data to: {OUT_PATH}")
