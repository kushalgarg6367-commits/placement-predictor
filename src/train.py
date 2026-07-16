"""
train.py
--------
Trains a baseline Logistic Regression and an XGBoost classifier on the
placement dataset, compares them, and generates SHAP explainability plots
for the final model. Saves the trained model + feature list for app.py.

Run:
    python src/train.py
"""

import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    MODEL_BACKEND = "xgboost"
except ImportError:
    # Falls back gracefully if xgboost isn't installed -- GradientBoosting
    # is a close sklearn equivalent so the rest of the pipeline (SHAP, app.py)
    # doesn't need to change.
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    MODEL_BACKEND = "sklearn-gradient-boosting (xgboost not found, used fallback)"

from preprocess import load_train_test, FEATURE_COLUMNS


def train():
    X_train, X_test, y_train, y_test = load_train_test()

    # ---- Baseline ----
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"[Baseline] Logistic Regression accuracy: {baseline_acc:.3f}")

    # ---- Main model ----
    print(f"Training main model using backend: {MODEL_BACKEND}")
    if MODEL_BACKEND == "xgboost":
        model = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "backend": MODEL_BACKEND,
        "baseline_logreg_accuracy": round(float(baseline_acc), 4),
        "model_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "model_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "model_f1": round(float(f1_score(y_test, preds)), 4),
    }
    print("Main model metrics:", json.dumps(metrics, indent=2))
    print("\nClassification report:\n", classification_report(y_test, preds))

    # ---- Confusion matrix plot ----
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Placed", "Placed"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Placed", "Placed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=150)
    plt.close()

    # ---- SHAP explainability ----
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig("assets/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved SHAP summary plot to assets/shap_summary.png")
    except ImportError:
        print("shap not installed -- skipping SHAP plot generation. "
              "Run `pip install shap` to enable explainability (used in app.py too).")

    # ---- Save artifacts ----
    joblib.dump(model, "models/placement_model.pkl")
    with open("models/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model to models/placement_model.pkl")
    return model, metrics


if __name__ == "__main__":
    train()
