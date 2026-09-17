"""
Feature Engineering module for Wine Quality dataset.

Transforms raw input features into an enriched feature representation
for model training and real-time inference.
"""

import pandas as pd
import numpy as np

RAW_FEATURE_COLUMNS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol"
]

ENGINEERED_FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + [
    "free_to_total_sulfur_ratio",
    "bound_sulfur_dioxide",
    "total_acidity",
    "acid_to_sugar_ratio"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific feature representations from raw wine measurements.

    Engineered features:
    1. free_to_total_sulfur_ratio: Ratio of free SO2 to total SO2 (wine stability indicator).
    2. bound_sulfur_dioxide: Total SO2 minus free SO2.
    3. total_acidity: Sum of fixed, volatile, and citric acid.
    4. acid_to_sugar_ratio: Balance between total acidity and residual sweetness.
    """
    df_feat = pd.DataFrame(index=df.index)

    for col in RAW_FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: '{col}'")
        df_feat[col] = df[col].astype(float)

    # 1. Sulfur dioxide dynamics
    df_feat["free_to_total_sulfur_ratio"] = df_feat["free_sulfur_dioxide"] / (df_feat["total_sulfur_dioxide"] + 1e-5)
    df_feat["bound_sulfur_dioxide"] = df_feat["total_sulfur_dioxide"] - df_feat["free_sulfur_dioxide"]

    # 2. Acidity balance
    df_feat["total_acidity"] = df_feat["fixed_acidity"] + df_feat["volatile_acidity"] + df_feat["citric_acid"]
    df_feat["acid_to_sugar_ratio"] = df_feat["total_acidity"] / (df_feat["residual_sugar"] + 1e-5)

    return df_feat[ENGINEERED_FEATURE_COLUMNS]
