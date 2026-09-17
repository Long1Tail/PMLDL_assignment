import argparse
import json
import logging
import os
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from features import (
        RAW_FEATURE_COLUMNS,
        ENGINEERED_FEATURE_COLUMNS,
        engineer_features,
    )
except ImportError:
    from code.models.features import (
        RAW_FEATURE_COLUMNS,
        ENGINEERED_FEATURE_COLUMNS,
        engineer_features,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ModelEngineering")


def build_pipeline(n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced"
        ))
    ])
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, float], np.ndarray]:
    """Calculate comprehensive evaluation metrics on test features."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    return metrics, y_pred


def plot_and_save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path):
    """Generate and save confusion matrix figure for MLflow artifacts."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Standard (<6)", "Good (>=6)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_model_engineering(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    model_output_dir: str = "models",
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    experiment_name: str = "wine_quality_pipeline"
):
    logger.info("=== Starting Stage 2: Model Engineering ===")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info("Loaded training data: %d rows, test data: %d rows", len(train_df), len(test_df))

    logger.info("Running feature engineering routines...")
    X_train_fe = engineer_features(train_df[RAW_FEATURE_COLUMNS])
    y_train = train_df["is_good_quality"]

    X_test_fe = engineer_features(test_df[RAW_FEATURE_COLUMNS])
    y_test = test_df["is_good_quality"]

    logger.info("Engineered feature matrix: %d features (%s)", len(ENGINEERED_FEATURE_COLUMNS), ENGINEERED_FEATURE_COLUMNS)

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_db_path = Path("mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment(experiment_name)

    model_dir = Path(model_output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("Started MLflow run: %s (Experiment: %s)", run_id, experiment_name)

        params = {
            "model_type": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "raw_features_count": len(RAW_FEATURE_COLUMNS),
            "engineered_features_count": len(ENGINEERED_FEATURE_COLUMNS)
        }
        mlflow.log_params(params)

        logger.info("Training model pipeline...")
        pipeline = build_pipeline(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        pipeline.fit(X_train_fe, y_train)

        logger.info("Evaluating model on test features...")
        metrics, y_pred = evaluate_model(pipeline, X_test_fe, y_test)
        logger.info("Evaluation Metrics: %s", metrics)
        mlflow.log_metrics(metrics)

        cm_path = model_dir / "confusion_matrix.png"
        plot_and_save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            serialization_format="cloudpickle"
        )

        model_file_path = model_dir / "model.joblib"
        joblib.dump(pipeline, model_file_path)
        logger.info("Packaged model saved to: %s", model_file_path)

        metadata = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "model_file": str(model_file_path.name),
            "raw_feature_columns": RAW_FEATURE_COLUMNS,
            "engineered_feature_columns": ENGINEERED_FEATURE_COLUMNS,
            "metrics": metrics,
            "params": params
        }
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
        logger.info("Model metadata saved to: %s", metadata_path)

    logger.info("=== Stage 2: Model Engineering completed successfully ===")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Model Engineering Pipeline")
    parser.add_argument("--train-path", default="data/processed/train.csv", help="Path to train CSV")
    parser.add_argument("--test-path", default="data/processed/test.csv", help="Path to test CSV")
    parser.add_argument("--model-output-dir", default="models", help="Folder to save packaged model")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=10, help="Max depth of trees")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-name", default="wine_quality_pipeline", help="MLflow experiment name")
    args = parser.parse_args()

    run_model_engineering(
        train_path=args.train_path,
        test_path=args.test_path,
        model_output_dir=args.model_output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        experiment_name=args.experiment_name
    )
