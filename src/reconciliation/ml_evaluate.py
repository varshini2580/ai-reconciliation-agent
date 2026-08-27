from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RECON_DIR = DATA_DIR / "reconciliation"

BINARY_TEST_FILE = RECON_DIR / "ml_binary_test.csv"
MULTICLASS_TEST_FILE = RECON_DIR / "ml_multiclass_test.csv"

BINARY_MODEL_FILE = RECON_DIR / "binary_model.joblib"
MULTICLASS_MODEL_FILE = RECON_DIR / "multiclass_model.joblib"

BINARY_RESULTS_FILE = RECON_DIR / "ml_binary_evaluation.csv"
MULTICLASS_RESULTS_FILE = RECON_DIR / "ml_multiclass_evaluation.csv"
FEATURE_IMPORTANCE_FILE = RECON_DIR / "ml_feature_importance.csv"


# ============================================================
# HELPERS
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

    return df[feature_columns].copy()


def load_model(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    return joblib.load(path)


# ============================================================
# BINARY EVALUATION
# ============================================================

def evaluate_binary():

    print()
    print("=" * 70)
    print("        BINARY MODEL EVALUATION")
    print("=" * 70)

    test_df = pd.read_csv(
        BINARY_TEST_FILE
    )

    pipeline = load_model(
        BINARY_MODEL_FILE
    )

    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]

    X_test = prepare_features(
        test_df,
        "target_exception",
    )

    y_test = test_df[
        "target_exception"
    ]

    X_processed = preprocessor.transform(
        X_test
    )

    predictions = model.predict(
        X_processed
    )

    probabilities = model.predict_proba(
        X_processed
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
    print("Binary evaluation results:")
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

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print()
    print("Confusion matrix:")
    print(
        "                 Predicted"
    )
    print(
        "                 MATCH  EXCEPTION"
    )
    print(
        f"Actual MATCH      {cm[0][0]:5d}  {cm[0][1]:9d}"
    )
    print(
        f"Actual EXCEPTION  {cm[1][0]:5d}  {cm[1][1]:9d}"
    )

    # --------------------------------------------------------
    # MISCLASSIFIED TRANSACTIONS
    # --------------------------------------------------------

    incorrect = test_df[
        predictions != y_test.values
    ].copy()

    incorrect["predicted_exception"] = (
        predictions[predictions != y_test.values]
    )

    incorrect["exception_probability"] = (
        probabilities[
            predictions != y_test.values,
            1
        ]
    )

    print()
    print(
        f"Binary misclassifications: "
        f"{len(incorrect)}"
    )

    if len(incorrect):

        print()
        print(
            "Binary misclassified transactions:"
        )

        print(
            incorrect[
                [
                    "transaction_id",
                    "target_exception",
                    "predicted_exception",
                    "target_exception_type",
                    "exception_probability",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result = test_df[
        [
            "transaction_id",
            "target_exception",
            "target_exception_type",
        ]
    ].copy()

    result["predicted_exception"] = predictions
    result["exception_probability"] = probabilities[:, 1]

    result["correct"] = (
        result["target_exception"]
        == result["predicted_exception"]
    )

    result.to_csv(
        BINARY_RESULTS_FILE,
        index=False,
    )

    print()
    print(
        f"Binary evaluation saved to:\n"
        f"{BINARY_RESULTS_FILE}"
    )

    return (
        test_df,
        predictions,
        probabilities,
    )


# ============================================================
# MULTICLASS EVALUATION
# ============================================================

def evaluate_multiclass():

    print()
    print("=" * 70)
    print("        MULTICLASS MODEL EVALUATION")
    print("=" * 70)

    test_df = pd.read_csv(
        MULTICLASS_TEST_FILE
    )

    pipeline = load_model(
        MULTICLASS_MODEL_FILE
    )

    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]

    X_test = prepare_features(
        test_df,
        "target_exception_type",
    )

    y_test = test_df[
        "target_exception_type"
    ]

    X_processed = preprocessor.transform(
        X_test
    )

    predictions = model.predict(
        X_processed
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print(
        f"Multiclass Accuracy: "
        f"{accuracy:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print()
    print("========== CLASSIFICATION REPORT ==========")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    print(report)

    # --------------------------------------------------------
    # MACRO / WEIGHTED METRICS
    # --------------------------------------------------------

    (
        precision_macro,
        recall_macro,
        f1_macro,
        _,
    ) = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _,
    ) = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(
        f"Macro Precision:    {precision_macro:.4f}"
    )
    print(
        f"Macro Recall:       {recall_macro:.4f}"
    )
    print(
        f"Macro F1:           {f1_macro:.4f}"
    )

    print(
        f"Weighted Precision: {precision_weighted:.4f}"
    )
    print(
        f"Weighted Recall:    {recall_weighted:.4f}"
    )
    print(
        f"Weighted F1:        {f1_weighted:.4f}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    labels = sorted(
        set(y_test) | set(predictions)
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[
            f"ACTUAL_{label}"
            for label in labels
        ],
        columns=[
            f"PREDICTED_{label}"
            for label in labels
        ],
    )

    print()
    print(
        "========== CONFUSION MATRIX =========="
    )

    print(
        confusion_df.to_string()
    )

    # --------------------------------------------------------
    # MISCLASSIFICATIONS
    # --------------------------------------------------------

    incorrect_mask = (
        predictions
        != y_test.values
    )

    incorrect = test_df[
        incorrect_mask
    ].copy()

    incorrect["predicted_exception_type"] = (
        predictions[incorrect_mask]
    )

    print()
    print(
        f"Multiclass misclassifications: "
        f"{len(incorrect)}"
    )

    if len(incorrect):

        print()
        print(
            "========== MISCLASSIFIED TRANSACTIONS =========="
        )

        print(
            incorrect[
                [
                    "transaction_id",
                    "target_exception_type",
                    "predicted_exception_type",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result = test_df[
        [
            "transaction_id",
            "target_exception_type",
        ]
    ].copy()

    result["predicted_exception_type"] = (
        predictions
    )

    result["correct"] = (
        result["target_exception_type"]
        == result["predicted_exception_type"]
    )

    result.to_csv(
        MULTICLASS_RESULTS_FILE,
        index=False,
    )

    print()
    print(
        f"Multiclass evaluation saved to:\n"
        f"{MULTICLASS_RESULTS_FILE}"
    )

    return (
        test_df,
        predictions,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance():

    print()
    print("=" * 70)
    print("        FEATURE IMPORTANCE")
    print("=" * 70)

    pipeline = load_model(
        BINARY_MODEL_FILE
    )

    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]

    # Get transformed feature names.
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "Top 20 binary-model features:"
    )

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print()
    print(
        f"Feature importance saved to:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        print("=" * 70)
        print("        PHASE 5.5 — ML EVALUATION")
        print("=" * 70)

        evaluate_binary()

        evaluate_multiclass()

        save_feature_importance()

        print()
        print("=" * 70)
        print("       PHASE 5.5 COMPLETED")
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print("       PHASE 5.5 FAILED")
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()