import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("DataEngineering")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by replacing spaces with underscores and lowercasing."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    """Load raw dataset from CSV file supporting comma or semicolon delimiters."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")

    logger.info("Loading raw data from %s", file_path)
    try:
        df = pd.read_csv(file_path, sep=";")
        if len(df.columns) <= 1:
            df = pd.read_csv(file_path, sep=",")
    except Exception as e:
        logger.error("Failed reading CSV: %s", e)
        raise

    df = normalize_column_names(df)
    logger.info("Loaded raw dataset shape: %s with columns: %s", df.shape, list(df.columns))
    return df


def clean_data(df: pd.DataFrame, iqr_multiplier: float = 2.5):
    df_clean = df.copy()

    missing_counts = df_clean.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning("Found missing values:\n%s", missing_counts[missing_counts > 0])
        # Impute numeric columns with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                logger.info("Imputed missing values in '%s' with median %f", col, median_val)
    else:
        logger.info("No missing values found in dataset.")

    feature_cols = [c for c in df_clean.columns if c != "quality"]
    initial_rows = len(df_clean)

    outlier_mask = pd.Series(False, index=df_clean.index)
    for col in feature_cols:
        q25 = df_clean[col].quantile(0.25)
        q75 = df_clean[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - (iqr_multiplier * iqr)
        upper_bound = q75 + (iqr_multiplier * iqr)
        col_outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        outlier_mask = outlier_mask | col_outliers

    df_clean = df_clean[~outlier_mask].reset_index(drop=True)
    removed_count = initial_rows - len(df_clean)
    logger.info(
        "Outlier removal (IQR multiplier=%.1f): removed %d rows (%.2f%%). Remaining rows: %d",
        iqr_multiplier, removed_count, (removed_count / initial_rows) * 100, len(df_clean)
    )

    df_clean["is_good_quality"] = (df_clean["quality"] >= 6).astype(int)
    logger.info(
        "Target distribution ('is_good_quality'):\n%s",
        df_clean["is_good_quality"].value_counts(normalize=True).to_dict()
    )

    return df_clean


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Splitting dataset: test_size=%.2f, random_state=%d", test_size, random_state)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["is_good_quality"]
    )
    logger.info("Train shape: %s, Test shape: %s", train_df.shape, test_df.shape)
    return train_df, test_df


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str | Path):
    """Save train and test sets to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved train data to %s (%d rows)", train_path, len(train_df))
    logger.info("Saved test data to %s (%d rows)", test_path, len(test_df))


def run_data_engineering(
    raw_path: str = "data/raw/winequality-red.csv",
    processed_dir: str = "data/processed",
    test_size: float = 0.2,
    iqr_multiplier: float = 2.5,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=== Starting Stage 1: Data Engineering ===")
    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw, iqr_multiplier=iqr_multiplier)
    train_df, test_df = split_data(df_clean, test_size=test_size, random_state=random_state)
    save_processed_data(train_df, test_df, processed_dir)
    logger.info("=== Stage 1: Data Engineering completed successfully ===")
    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Data Engineering Pipeline")
    parser.add_argument("--raw-path", default="data/raw/winequality-red.csv", help="Path to raw CSV file")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to output processed folder")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--iqr-multiplier", type=float, default=2.5, help="IQR outlier multiplier")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    run_data_engineering(
        raw_path=args.raw_path,
        processed_dir=args.processed_dir,
        test_size=args.test_size,
        iqr_multiplier=args.iqr_multiplier,
        random_state=args.random_state
    )
