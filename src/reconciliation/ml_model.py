from pathlib import Path
import sys

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RECON_DIR = DATA_DIR / "reconciliation"

BINARY_TRAIN_FILE = RECON_DIR / "ml_binary_train.csv"
BINARY_TEST_FILE = RECON_DIR / "ml_binary_test.csv"

MULTICLASS_TRAIN_FILE = RECON_DIR / "ml_multiclass_train.csv"
MULTICLASS_TEST_FILE = RECON_DIR / "ml_multiclass_test.csv"

BINARY_MODEL_FILE = RECON_DIR / "binary_model.joblib"
MULTICLASS_MODEL_FILE = RECON_DIR / "multiclass_model.joblib"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 200


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("        PHASE 5.4 — ML MODEL TRAINING")
    print("=" * 70)

    files = [
        BINARY_TRAIN_FILE,
        BINARY_TEST_FILE,
        MULTICLASS_TRAIN_FILE,
        MULTICLASS_TEST_FILE,
    ]

    for path in files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    binary_train = pd.read_csv(
        BINARY_TRAIN_FILE
    )

    binary_test = pd.read_csv(
        BINARY_TEST_FILE
    )

    multiclass_train = pd.read_csv(
        MULTICLASS_TRAIN_FILE
    )

    multiclass_test = pd.read_csv(
        MULTICLASS_TEST_FILE
    )

    print(
        f"Binary train records: {len(binary_train)}"
    )

    print(
        f"Binary test records: {len(binary_test)}"
    )

    print(
        f"Multiclass train records: "
        f"{len(multiclass_train)}"
    )

    print(
        f"Multiclass test records: "
        f"{len(multiclass_test)}"
    )

    print("[OK] Training data loaded")

    return (
        binary_train,
        binary_test,
        multiclass_train,
        multiclass_test,
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(X):

    categorical_columns = [
        column
        for column in X.columns
        if X[column].dtype == "object"
    ]

    numerical_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns,
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns,
            ),
        ]
    )

    return transformer


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df, target_column):

    excluded = {
        "transaction_id",
        "target_exception",
        "target_exception_type",
        target_column,
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    if not feature_columns:

        raise ValueError(
            "No ML features available."
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    return X, y


# ============================================================
# BINARY MODEL
# ============================================================

def train_binary_model(train_df, test_df):

    print()
    print("=" * 70)
    print("        BINARY MODEL")
    print("=" * 70)

    X_train, y_train = prepare_features(
        train_df,
        "target_exception",
    )

    X_test, y_test = prepare_features(
        test_df,
        "target_exception",
    )

    print(
        f"Features used: {X_train.shape[1]}"
    )

    preprocessor = create_preprocessor(
        X_train
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    print("Training Random Forest...")

    model.fit(
        X_train_processed,
        y_train,
    )

    predictions = model.predict(
        X_test_processed
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    print()
    print("Binary model results:")
    print(
        f"Accuracy:  {accuracy:.4f}"
    )
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall:    {recall:.4f}"
    )
    print(
        f"F1 Score:  {f1:.4f}"
    )

    return (
        preprocessor,
        model,
    )


# ============================================================
# MULTICLASS MODEL
# ============================================================

def train_multiclass_model(
    train_df,
    test_df,
):

    print()
    print("=" * 70)
    print("        MULTICLASS MODEL")
    print("=" * 70)

    X_train, y_train = prepare_features(
        train_df,
        "target_exception_type",
    )

    X_test, y_test = prepare_features(
        test_df,
        "target_exception_type",
    )

    print(
        f"Features used: {X_train.shape[1]}"
    )

    print(
        f"Classes: {y_train.nunique()}"
    )

    preprocessor = create_preprocessor(
        X_train
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    print("Training Random Forest...")

    model.fit(
        X_train_processed,
        y_train,
    )

    predictions = model.predict(
        X_test_processed
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    print()
    print("Multiclass model results:")

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        f"Macro Precision:   {precision_macro:.4f}"
    )

    print(
        f"Macro Recall:      {recall_macro:.4f}"
    )

    print(
        f"Macro F1:          {f1_macro:.4f}"
    )

    print(
        f"Weighted Precision:{precision_weighted:.4f}"
    )

    print(
        f"Weighted Recall:   {recall_weighted:.4f}"
    )

    print(
        f"Weighted F1:       {f1_weighted:.4f}"
    )

    return (
        preprocessor,
        model,
    )


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(
    binary_preprocessor,
    binary_model,
    multiclass_preprocessor,
    multiclass_model,
):

    import joblib

    binary_pipeline = {
        "preprocessor": binary_preprocessor,
        "model": binary_model,
    }

    multiclass_pipeline = {
        "preprocessor": multiclass_preprocessor,
        "model": multiclass_model,
    }

    joblib.dump(
        binary_pipeline,
        BINARY_MODEL_FILE,
    )

    joblib.dump(
        multiclass_pipeline,
        MULTICLASS_MODEL_FILE,
    )

    print()
    print("Models saved:")

    print(
        BINARY_MODEL_FILE
    )

    print(
        MULTICLASS_MODEL_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        (
            binary_train,
            binary_test,
            multiclass_train,
            multiclass_test,
        ) = load_data()

        (
            binary_preprocessor,
            binary_model,
        ) = train_binary_model(
            binary_train,
            binary_test,
        )

        (
            multiclass_preprocessor,
            multiclass_model,
        ) = train_multiclass_model(
            multiclass_train,
            multiclass_test,
        )

        save_models(
            binary_preprocessor,
            binary_model,
            multiclass_preprocessor,
            multiclass_model,
        )

        print()
        print("=" * 70)
        print("       PHASE 5.4 COMPLETED")
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print("       PHASE 5.4 FAILED")
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()