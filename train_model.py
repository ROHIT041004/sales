"""
train_model.py
----------------
Trains the "Item Risk Scoring" model used by the StockSense AI dashboard.

WHY A SYNTHETIC LABEL?
The BigMart-style dataset (bigmart_with_subtypes.csv) has no ground-truth
"stockout risk" column, so we engineer one that is realistic and explainable:

    HIGH RISK (label = 1)  ->  an item that historically SELLS WELL
                                (top third of Item_Outlet_Sales for its
                                Item_Type) but is given LOW SHELF VISIBILITY
                                (bottom third of Item_Visibility for its
                                Item_Type).
                                => strong demand + poor shelf exposure is the
                                   classic recipe for running out of stock.

    LOW RISK  (label = 0)  ->  everything else.

This mirrors real retail practice: planners flag "high-demand / low-visibility"
SKUs for reorder review. You can swap this out for a real historical
stockout/backorder column the moment you have one -- just change
`build_risk_label()` below and re-run this script.

Run:
    python train_model.py
Outputs:
    risk_model.pkl        (trained sklearn Pipeline)
    feature_importance.json
    model_metrics.json
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "bigmart_with_subtypes.csv"
MODEL_PATH = "risk_model.pkl"

NUMERIC_FEATURES = ["Item_Weight", "Item_MRP", "Item_Visibility"]
CATEGORICAL_FEATURES = [
    "Item_Fat_Content",
    "Item_Type",
    "Item_Sub_Type",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}
    )
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown")
    return df


def build_risk_label(df: pd.DataFrame) -> pd.Series:
    """High demand (top third of sales) + low visibility (bottom third) -> risk=1."""
    sales_rank = df.groupby("Item_Type")["Item_Outlet_Sales"].rank(pct=True)
    vis_rank = df.groupby("Item_Type")["Item_Visibility"].rank(pct=True)
    high_demand = sales_rank >= 0.66
    low_visibility = vis_rank <= 0.33
    return (high_demand & low_visibility).astype(int)


def main():
    df = load_data()
    df["risk_label"] = build_risk_label(df)
    print("Risk label distribution:\n", df["risk_label"].value_counts())

    X = df[ALL_FEATURES]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest AUC: {auc:.3f}  |  Test Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

    # ---- feature importance (grouped back to the original field names) ----
    ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    all_names = NUMERIC_FEATURES + list(cat_names)
    importances = pipe.named_steps["model"].feature_importances_

    grouped = {f: 0.0 for f in ALL_FEATURES}
    for name, imp in zip(all_names, importances):
        base = name.split("_")[0] + "_" + name.split("_")[1] if "_" in name else name
        matched = next((f for f in ALL_FEATURES if name.startswith(f)), None)
        grouped[matched if matched else name] = grouped.get(matched if matched else name, 0.0) + imp

    total = sum(grouped.values()) or 1.0
    grouped_pct = {k: round(100 * v / total, 1) for k, v in grouped.items()}
    grouped_pct = dict(sorted(grouped_pct.items(), key=lambda x: -x[1]))

    joblib.dump(pipe, MODEL_PATH)
    with open("feature_importance.json", "w") as f:
        json.dump(grouped_pct, f, indent=2)
    with open("model_metrics.json", "w") as f:
        json.dump({"test_auc": round(auc, 4), "test_accuracy": round(acc, 4), "n_rows": len(df)}, f, indent=2)

    print(f"\nSaved: {MODEL_PATH}, feature_importance.json, model_metrics.json")
    print("Top signals:", list(grouped_pct.items())[:5])


if __name__ == "__main__":
    main()
