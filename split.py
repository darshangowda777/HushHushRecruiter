import os
import pandas as pd
import numpy as np
from pathlib import Path

csv_path = "/Users/darshan/Desktop/hushhhhhh/github_data_labeled_smoted.csv"
assert Path(csv_path).exists(), "Could not find the uploaded CSV at /mnt/data/github_data_labeled_smoted.csv"

df = pd.read_csv(csv_path)
n_rows, n_cols = df.shape

preferred_names = [
    "label","Label","target","Target","class","Class",
    "is_bug","bug","is_buggy","defect","is_defective","is_vulnerable",
    "vulnerable","is_malicious","malicious","spam","is_spam","category","Category","y"
]

target_col = None

for name in preferred_names:
    if name in df.columns:
        target_col = name
        break

def looks_like_id(colname: str) -> bool:
    return "id" in colname.lower() or colname.lower() in {"index","idx"}

if target_col is None:
    candidates = []
    for col in df.columns:
        if looks_like_id(col):
            continue
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique <= 5:  
            candidates.append((col, nunique, str(df[col].dtype)))
    if candidates:
        candidates = sorted(candidates, key=lambda x: (x[1], x[0]))
        target_col = candidates[0][0]

if target_col is None:
    last_col = df.columns[-1]
    if not looks_like_id(last_col) and df[last_col].nunique(dropna=True) > 1:
        target_col = last_col

use_stratify = target_col is not None

rng = np.random.RandomState(42)

if use_stratify:
    test_indices = []
    for cls_value, group in df.groupby(target_col):
        n = len(group)
        if n == 0:
            continue
        n_test = max(1, int(round(0.2 * n)))
        if n == 1:
            continue
        sampled = group.sample(n=n_test, random_state=rng)
        test_indices.append(sampled.index)

    if test_indices:
        test_idx = pd.Index(sorted(np.concatenate(test_indices)))
        test = df.loc[test_idx]
        train = df.drop(index=test_idx)
    else:
        perm = rng.permutation(n_rows)
        test_size = int(round(0.2 * n_rows))
        test = df.iloc[perm[:test_size]]
        train = df.iloc[perm[test_size:]]
        use_stratify = False
else:
    perm = rng.permutation(n_rows)
    test_size = int(round(0.2 * n_rows))
    test = df.iloc[perm[:test_size]]
    train = df.iloc[perm[test_size:]]

train_path = "/Users/darshan/Desktop/hushhhhhh/train.csv"
test_path = "/Users/darshan/Desktop/hushhhhhh/test.csv"
train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

summary_rows = []

if use_stratify:
    full_counts = df[target_col].value_counts(dropna=False)
    train_counts = train[target_col].value_counts(dropna=False)
    test_counts = test[target_col].value_counts(dropna=False)

    all_classes = sorted(full_counts.index.tolist(), key=lambda x: str(x))
    for cls in all_classes:
        total = full_counts.get(cls, 0)
        trn = train_counts.get(cls, 0)
        tst = test_counts.get(cls, 0)
        summary_rows.append({
            f"{target_col}": cls,
            "Total": total,
            "Train": trn,
            "Test": tst,
            "Train %": (trn/len(train))*100 if len(train) else np.nan,
            "Test %": (tst/len(test))*100 if len(test) else np.nan,
            "Overall %": (total/len(df))*100 if len(df) else np.nan
        })

    summary_df = pd.DataFrame(summary_rows)
else:
    summary_df = pd.DataFrame({
        "Info": ["Stratified split not used (target not found)"],
        "Rows": [len(df)]
    })

