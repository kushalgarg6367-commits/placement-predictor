"""
preprocess.py
-------------
Cleaning + feature encoding for the placement dataset.
Kept as a separate module (not stuffed into train.py) so app.py can
import and reuse the exact same transformation at inference time --
this avoids the classic "training/serving skew" bug.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

BRANCH_ORDER = ["CSE", "ECE", "Electrical", "Mechanical", "Chemical", "Civil"]
FEATURE_COLUMNS = [
    "cgpa", "internships", "projects_completed", "backlogs",
    "coding_score", "communication_score", "certifications",
    "extracurricular", "branch_CSE", "branch_ECE", "branch_Electrical",
    "branch_Mechanical", "branch_Chemical", "branch_Civil",
]
TARGET_COLUMN = "placed"


def load_raw(path="data/placement_data.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Median imputation for the two numeric columns we injected NaNs into.
    # Median > mean here because scores are slightly skewed by branch bias.
    for col in ["coding_score", "communication_score"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode branch. Uses a fixed category list so a single-row
    inference dataframe (from the Streamlit form) always produces the same
    columns as training, even if that row's branch isn't every category."""
    df = df.copy()
    df["branch"] = pd.Categorical(df["branch"], categories=BRANCH_ORDER)
    df = pd.get_dummies(df, columns=["branch"], prefix="branch")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0  # branch category not present in this batch

    return df


def prepare_features(df: pd.DataFrame):
    df = clean(df)
    df = encode(df)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    return X, y


def load_train_test(path="data/placement_data.csv", test_size=0.2, seed=42):
    df = load_raw(path)
    X, y = prepare_features(df)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_train_test()
    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
    print("Train placement rate:", round(y_train.mean(), 3))
