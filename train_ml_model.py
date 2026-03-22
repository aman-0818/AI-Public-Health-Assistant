"""
ML Model Training Script
Disease Risk Prediction — Dengue, Malaria, Typhoid
Uses: XGBoost + SMOTE + Synthetic Negatives
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "ml_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: LOAD & COMBINE ALL EXCEL DATA
# ─────────────────────────────────────────────
def load_positive_cases():
    print("\n" + "="*60)
    print("STEP 1: Loading positive case data...")
    print("="*60)

    disease_map = {
        "Dengue":  ["Dengue_IgM ELISA.samp.xlsx",
                    "Dengue_Rapid IgM Test.samp-2.xlsx",
                    "Dengue_Rapid NS1.samp-2.xlsx",
                    "Dengue_RT-PCR.samp-2.xlsx"],
        "Malaria": ["Malaria_Microscopy-Peripheral Blood Smear(MP).samp.xlsx",
                    "Malaria_Rapid Diagnostic Test(Malaria Ag).samp.xlsx"],
        "Typhoid": ["Typhoid_Cultures.samp.xlsx",
                    "Typhoid_TyphiDot-Rapid Card test.samp.xlsx"],
    }

    frames = []
    for disease, files in disease_map.items():
        for fname in files:
            fpath = os.path.join(DATASET_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  [SKIP] Not found: {fname}")
                continue
            df = pd.read_excel(fpath)
            df["disease_label"] = disease
            frames.append(df)
            print(f"  Loaded {len(df):>5} rows  <- {fname}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total positive cases: {len(combined)}")
    print(f"  Distribution:\n{combined['disease_label'].value_counts().to_string()}")
    return combined


# ─────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    print("\n" + "="*60)
    print("STEP 2: Feature engineering...")
    print("="*60)

    out = pd.DataFrame()

    # Latitude & Longitude
    out["latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    out["longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # Age — extract numeric value ("32 Years" → 32)
    out["age"] = df["Age"].astype(str).str.extract(r"(\d+)").astype(float)

    # Gender → 0/1
    out["gender"] = df["Gender"].astype(str).str.lower().map(
        {"male": 1, "female": 0}
    )

    # Month & Season from Date Of Onset
    date_col = None
    for col in ["Date Of Onset", "Sample Collected Date"]:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        out["month"] = dates.dt.month
        # Season: 1=Summer(Mar-May), 2=Monsoon(Jun-Sep), 3=Winter(Oct-Feb)
        out["season"] = out["month"].map(lambda m:
            1 if m in [3, 4, 5] else
            2 if m in [6, 7, 8, 9] else 3
        )
    else:
        out["month"]  = 7   # default monsoon
        out["season"] = 2

    # Disease label
    out["disease_label"] = df["disease_label"].values

    # Drop rows with missing lat/lon/age
    before = len(out)
    out = out.dropna(subset=["latitude", "longitude", "age"])
    after = len(out)
    if before != after:
        print(f"  Dropped {before - after} rows with missing lat/lon/age")

    # Fill remaining nulls
    out["gender"] = out["gender"].fillna(0.5)
    out["month"]  = out["month"].fillna(7)
    out["season"] = out["season"].fillna(2)

    print(f"  Features ready: {len(out)} rows × {len(out.columns)} columns")
    print(f"  Columns: {list(out.columns)}")
    return out


# ─────────────────────────────────────────────
# STEP 3: GENERATE SYNTHETIC NEGATIVE CASES
# ─────────────────────────────────────────────
def generate_negatives(positive_df, ratio=0.4):
    """
    Create synthetic 'No Disease' samples.
    Strategy: Sample random locations from India's Tamil Nadu bounding box
    that are NOT near any existing positive case.
    """
    print("\n" + "="*60)
    print("STEP 3: Generating synthetic negative cases...")
    print("="*60)

    # Tamil Nadu approximate bounding box
    LAT_MIN, LAT_MAX = 8.0,  13.6
    LON_MIN, LON_MAX = 76.2, 80.4

    n_neg = int(len(positive_df) * ratio)
    np.random.seed(42)

    neg_rows = []
    attempts = 0
    while len(neg_rows) < n_neg and attempts < n_neg * 10:
        attempts += 1
        lat = np.random.uniform(LAT_MIN, LAT_MAX)
        lon = np.random.uniform(LON_MIN, LON_MAX)

        # Only keep if no positive case within 0.05 degrees (~5 km)
        dists = np.sqrt(
            (positive_df["latitude"]  - lat) ** 2 +
            (positive_df["longitude"] - lon) ** 2
        )
        if dists.min() > 0.05:
            neg_rows.append({
                "latitude":      lat,
                "longitude":     lon,
                "age":           np.random.randint(5, 75),
                "gender":        np.random.choice([0, 1]),
                "month":         np.random.randint(1, 13),
                "season":        np.random.choice([1, 2, 3]),
                "disease_label": "Negative",
            })

    neg_df = pd.DataFrame(neg_rows)
    print(f"  Generated {len(neg_df)} synthetic negative cases")
    return neg_df


# ─────────────────────────────────────────────
# STEP 4: SMOTE — Augment Minority Classes
# ─────────────────────────────────────────────
def apply_smote(X, y, target_counts):
    print("\n" + "="*60)
    print("STEP 4: Applying SMOTE for minority classes...")
    print("="*60)

    print("  Before SMOTE:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Class {u}: {c}")

    smote = SMOTE(sampling_strategy=target_counts, random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X, y)

    print("\n  After SMOTE:")
    unique, counts = np.unique(y_res, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Class {u}: {c}")

    return X_res, y_res


# ─────────────────────────────────────────────
# STEP 5: TRAIN XGBOOST
# ─────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, y_test, label_names):
    print("\n" + "="*60)
    print("STEP 5: Training XGBoost model...")
    print("="*60)

    evals_result = {}

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric=["mlogloss", "merror"],
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )

    print("  Training with 300 rounds (early stopping at 20)...\n")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=25,
    )

    evals_result = model.evals_result()
    return model, evals_result


# ─────────────────────────────────────────────
# STEP 6: EVALUATE & PLOT
# ─────────────────────────────────────────────
def evaluate_and_plot(model, X_test, y_test, label_names, evals_result):
    print("\n" + "="*60)
    print("STEP 6: Evaluation & Plots...")
    print("="*60)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Best iteration: {model.best_iteration}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("XGBoost Disease Risk Model — Training Results", fontsize=14, fontweight="bold")

    # ── Plot 1: Training Loss (mlogloss) ──
    ax = axes[0]
    train_loss = evals_result["validation_0"]["mlogloss"]
    test_loss  = evals_result["validation_1"]["mlogloss"]
    rounds = range(len(train_loss))
    ax.plot(rounds, train_loss, label="Train Loss", color="royalblue")
    ax.plot(rounds, test_loss,  label="Val Loss",   color="tomato")
    ax.axvline(model.best_iteration, color="green", linestyle="--", label=f"Best round: {model.best_iteration}")
    ax.set_title("Training Loss (mlogloss)")
    ax.set_xlabel("Rounds (like epochs)")
    ax.set_ylabel("Log Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Plot 2: Confusion Matrix ──
    ax = axes[1]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # ── Plot 3: Feature Importance ──
    ax = axes[2]
    feature_names = ["latitude", "longitude", "age", "gender", "month", "season"]
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importance[sorted_idx],
        color="steelblue"
    )
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {plot_path}")
    plt.show()

    return acc, f1


# ─────────────────────────────────────────────
# STEP 7: SAVE MODEL
# ─────────────────────────────────────────────
def save_model(model, label_encoder, acc, f1):
    print("\n" + "="*60)
    print("STEP 7: Saving model...")
    print("="*60)

    model_path = os.path.join(OUTPUT_DIR, "disease_xgb_model.json")
    model.save_model(model_path)

    meta = {
        "trained_at":   datetime.now().isoformat(),
        "accuracy":     round(acc, 4),
        "f1_score":     round(f1, 4),
        "classes":      list(label_encoder.classes_),
        "features":     ["latitude", "longitude", "age", "gender", "month", "season"],
        "model_file":   "disease_xgb_model.json",
        "best_iteration": int(model.best_iteration),
    }
    meta_path = os.path.join(OUTPUT_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model saved : {model_path}")
    print(f"  Metadata    : {meta_path}")
    print(f"\n  Classes: {list(label_encoder.classes_)}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DISEASE RISK ML MODEL -- TRAINING PIPELINE")
    print("="*60)

    # Step 1: Load data
    raw_df = load_positive_cases()

    # Step 2: Feature engineering
    feat_df = engineer_features(raw_df)

    # Step 3: Synthetic negatives
    neg_df = generate_negatives(feat_df)

    # Combine positive + negative
    full_df = pd.concat([feat_df, neg_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Full dataset: {len(full_df)} rows")
    print(f"  Class distribution:\n{full_df['disease_label'].value_counts().to_string()}")

    # Encode labels
    le = LabelEncoder()
    full_df["label"] = le.fit_transform(full_df["disease_label"])
    label_names = list(le.classes_)
    print(f"\n  Label encoding: {dict(zip(label_names, le.transform(label_names)))}")

    # Features & target
    FEATURES = ["latitude", "longitude", "age", "gender", "month", "season"]
    X = full_df[FEATURES].values
    y = full_df["label"].values

    # Step 4: SMOTE on minority classes (before split to avoid data leakage we'll split first)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # Apply SMOTE only on training data
    class_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    max_count = max(class_counts.values())

    # Target: bring all classes to at least 1000 samples
    smote_target = {}
    for cls, cnt in class_counts.items():
        if cnt < 1000:
            smote_target[cls] = 1000

    if smote_target:
        X_train, y_train = apply_smote(X_train, y_train, smote_target)
    else:
        print("\n  SMOTE: All classes already have enough samples, skipping.")

    # Step 5: Train XGBoost
    model, evals_result = train_xgboost(X_train, X_test, y_train, y_test, label_names)

    # Step 6: Evaluate
    acc, f1 = evaluate_and_plot(model, X_test, y_test, label_names, evals_result)

    # Step 7: Save
    save_model(model, le, acc, f1)

    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print(f"  Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
