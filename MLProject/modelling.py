import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, log_loss
)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_confusion_matrix(cm, outpath: str):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="namadataset_preprocessing/loan_processed.csv")
    parser.add_argument("--experiment_name", type=str, default="CI_Retrain_Loan")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # Local tracking (CI nanti cukup upload folder mlruns/ atau outputs/)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment_name)

    df = pd.read_csv(args.data_path)

    # target encoding
    y = df["Loan_Status"]
    X = df.drop(columns=["Loan_Status"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000))
    ])

    with mlflow.start_run(run_name="CI_Retrain_Run") as run:
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        ll = log_loss(y_test, y_proba)

        mlflow.log_params({
            "model": "LogisticRegression",
            "scaler": "StandardScaler",
            "test_size": 0.2,
            "random_state": 42
        })

        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_roc_auc": auc,
            "test_log_loss": ll
        })

        # log model (dipakai untuk build-docker nanti)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        # ====== artifacts tambahan (untuk Skilled/Advance CI) ======
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "log_loss": ll}, f, indent=2)
        mlflow.log_artifact(metrics_path, artifact_path="reports")

        # simpan run_id agar workflow bisa build-docker dari run ini
        run_id_path = os.path.join(args.output_dir, "run_id.txt")
        with open(run_id_path, "w") as f:
            f.write(run.info.run_id)
        mlflow.log_artifact(run_id_path, artifact_path="meta")

        # simpan model lokal juga (opsional)
        local_model_path = os.path.join(args.output_dir, "model.joblib")
        joblib.dump(pipe, local_model_path)

        print("✅ CI retrain finished")
        print("Run ID:", run.info.run_id)
        print(f"Accuracy={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

if __name__ == "__main__":
    main()

#testing actions
