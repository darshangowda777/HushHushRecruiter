import os
import sys
import argparse
import urllib.parse as up
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score,average_precision_score,accuracy_score,confusion_matrix,classification_report)

# configuration
CSV_PATH        = "test.csv"      
TARGET          = "cluster"       
GOOD_VALUE      = 0              
ID_COL          = None           
OWNER_NAME      = "owner"         
EMAIL_NAME      = "email"        
N_SPLITS        = 5
RANDOM_STATE    = 42
TOP_K           = 25
SELECTION_MODE  = "predicted"    
PRED_THRESHOLD  = 0.42
DEDUP_EMAIL     = True
MAX_PER_OWNER   = None           

# push results to mysql
WRITE_TO_MYSQL  = True 
DB_HOST         = os.getenv("DB_HOST", "localhost")
DB_USER         = os.getenv("DB_USER", "root")
DB_PASS         = os.getenv("DB_PASS", "@darshu123")
DB_NAME         = os.getenv("DB_NAME", "hushhush")
DB_PORT         = int(os.getenv("DB_PORT", "3306"))
TABLE_NAME      = os.getenv("DB_TABLE", "top_25_candidates") 

def find_col_case_insensitive(columns, name):
    n = str(name).strip().lower()
    for c in columns:
        if str(c).strip().lower() == n:
            return c
    return None

def build_pipeline(num_cols, cat_cols):
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler(with_mean=False)),])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore")),])
    pre = ColumnTransformer([("num", num_tf, num_cols),("cat", cat_tf, cat_cols),])
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    return Pipeline([("prep", pre), ("clf", model)])

def drop_perfect_leakage(X, y):
    to_drop = []
    for c in X.columns:
        col = X[c]
        if col.isna().all():
            continue
        df_tmp = pd.DataFrame({"f": col, "y": y}).dropna()
        if df_tmp.empty:
            continue
        if df_tmp.groupby("f")["y"].nunique().max() == 1 and df_tmp["f"].nunique() >= 2:
            to_drop.append(c)
    if to_drop:
        X = X.drop(columns=to_drop)
    return X

def enforce_owner_cap(df_sorted, owner_col, max_per_owner, k):
    if owner_col is None or max_per_owner is None:
        return df_sorted.head(k)
    counts = {}
    rows = []
    for _, r in df_sorted.iterrows():
        owner = r[owner_col]
        counts[owner] = counts.get(owner, 0)
        if counts[owner] < max_per_owner:
            rows.append(r)
            counts[owner] += 1
        if len(rows) == k:
            break
    return pd.DataFrame(rows)

def train(csv_path=CSV_PATH,target=TARGET,good_value=GOOD_VALUE,id_col=ID_COL,owner_name=OWNER_NAME,email_name=EMAIL_NAME):
   
    df = pd.read_csv(csv_path)
    owner_col = find_col_case_insensitive(df.columns, owner_name)
    email_col = find_col_case_insensitive(df.columns, email_name)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns: {list(df.columns)}")

    df = df.dropna(subset=[target]).copy()
    y_raw = df[target]
    y = (y_raw.astype(str).str.strip().str.lower()
         == str(good_value).strip().lower()).astype(int)

    X = df.drop(columns=[target])

    non_feature_cols = []
    if id_col and id_col in X.columns: non_feature_cols.append(id_col)
    if owner_col and owner_col in X.columns: non_feature_cols.append(owner_col)
    if email_col and email_col in X.columns: non_feature_cols.append(email_col)
    X = X.drop(columns=non_feature_cols, errors="ignore")

    X = drop_perfect_leakage(X, y)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Build and fit pipeline on FULL data
    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X, y)
    return pipe

# Full script flow (CV, metrics, selection, optional DB write, model.pkl)
def run_full_flow(args):
    csv_path = args.csv
    df = pd.read_csv(csv_path)

    # Resolve columns
    owner_col = find_col_case_insensitive(df.columns, OWNER_NAME)
    email_col = find_col_case_insensitive(df.columns, EMAIL_NAME)
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not in columns: {list(df.columns)}")
    if owner_col is None:
        print(f"(Note) Owner column '{OWNER_NAME}' not found (case-insensitive).")
    if email_col is None:
        print(f"(Note) Email column '{EMAIL_NAME}' not found (case-insensitive).")

    # Target and features
    df = df.dropna(subset=[TARGET]).copy()
    y_raw = df[TARGET]
    y = (y_raw.astype(str).str.strip().str.lower()
         == str(GOOD_VALUE).strip().lower()).astype(int)

    X = df.drop(columns=[TARGET])

    # Don’t let IDs/owner/email leak into features
    non_feature_cols = []
    if ID_COL and ID_COL in X.columns: non_feature_cols.append(ID_COL)
    if owner_col and owner_col in X.columns: non_feature_cols.append(owner_col)
    if email_col and email_col in X.columns: non_feature_cols.append(email_col)
    X = X.drop(columns=non_feature_cols, errors="ignore")

    # Leakage guard
    X = drop_perfect_leakage(X, y)

    # Column typing
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # OOF probabilities
    oof_probs = np.zeros(len(X), dtype=float)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        pipe = build_pipeline(num_cols, cat_cols)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_probs[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        print(f"[Fold {fold}] done.")

    print(f"ROC AUC:           {roc_auc_score(y, oof_probs):.4f}")
    print(f"Average Precision: {average_precision_score(y, oof_probs):.4f}")

    # Build OOF-scored table with ID/owner/email
    if ID_COL and ID_COL in df.columns and df[ID_COL].is_unique:
        id_series = df[ID_COL]
        id_name = ID_COL
    else:
        id_series = pd.Series(df.index, name="row_index")
        id_name = "row_index"

    cols = {id_name: id_series.values, "pred_good_prob_oof": oof_probs}
    if owner_col: cols[owner_col] = df[owner_col].values
    if email_col: cols[email_col] = df[email_col].values
    all_oof = pd.DataFrame(cols)
    all_oof["y_true"] = y.values

    # Choose top K depending on selection mode
    if SELECTION_MODE.lower() == "predicted":
        predicted_good = all_oof.query("pred_good_prob_oof >= @PRED_THRESHOLD").copy()
        predicted_good = predicted_good.sort_values(
            ["pred_good_prob_oof", id_name], ascending=[False, True]
        )
        if DEDUP_EMAIL and email_col in predicted_good.columns:
            predicted_good = predicted_good.drop_duplicates(subset=[email_col], keep="first")
        if MAX_PER_OWNER is not None and owner_col in predicted_good.columns:
            top_25 = enforce_owner_cap(predicted_good, owner_col, MAX_PER_OWNER, TOP_K)
        else:
            top_25 = predicted_good.head(TOP_K)

    elif SELECTION_MODE.lower() == "actual":
        actual_good = all_oof.query("y_true == 1").copy()
        actual_good = actual_good.sort_values(
            ["pred_good_prob_oof", id_name], ascending=[False, True]
        )
        if DEDUP_EMAIL and email_col in actual_good.columns:
            actual_good = actual_good.drop_duplicates(subset=[email_col], keep="first")
        if MAX_PER_OWNER is not None and owner_col in actual_good.columns:
            top_25 = enforce_owner_cap(actual_good, owner_col, MAX_PER_OWNER, TOP_K)
        else:
            top_25 = actual_good.head(TOP_K)
    else:
        raise ValueError("SELECTION_MODE must be 'predicted' or 'actual'.")

    top_25 = top_25.reset_index(drop=True)

    # Accuracy from OOF predictions
    THRESH = PRED_THRESHOLD
    oof_pred = (oof_probs >= THRESH).astype(int)
    acc = accuracy_score(y, oof_pred)
    cm = confusion_matrix(y, oof_pred)
    print(f"\nAccuracy (threshold={THRESH:.4f}): {acc:.4f}")
    print("Confusion matrix ")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y, oof_pred, digits=4))

    # Optional: write (owner,email) to MySQL
    if args.write_mysql or WRITE_TO_MYSQL:
        if owner_col is None or email_col is None:
            raise ValueError(f"Could not find owner/email in top_25. Columns: {list(top_25.columns)}")
        to_push = top_25[[owner_col, email_col]].copy()
        to_push.columns = ["owner", "email"]

        encoded_pw = up.quote_plus(DB_PASS)

        from sqlalchemy import create_engine
        engine = create_engine(
            f"mysql+pymysql://{DB_USER}:{encoded_pw}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4",
            pool_pre_ping=True,
        )
        to_push.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
        print(f"[MySQL] Wrote {len(to_push)} rows (owner,email) to `{DB_NAME}`.`{TABLE_NAME}`.")

    
    print("\nTraining final model on full data and saving to model.pkl ...")
    model = train(csv_path=csv_path)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved model to model.pkl")


def parse_args():
    p = argparse.ArgumentParser(description="Supervised training, CV metrics, optional MySQL, and model export.")
    p.add_argument("--csv", default=CSV_PATH, help="Path to input CSV (default: %(default)s)")
    p.add_argument("--write-mysql", action="store_true", help="Write (owner,email) top-K to MySQL")
    return p.parse_args()

def main():
    args = parse_args()
    run_full_flow(args)

if __name__ == "__main__":
    main()
 