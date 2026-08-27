from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

RECONCILIATION_FILE = (
    BASE_DIR
    / "data"
    / "reconciliation"
    / "reconciliation_results.csv"
)

GROUND_TRUTH_FILE = (
    BASE_DIR
    / "data"
    / "ground_truth"
    / "ground_truth.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "reconciliation"
)

EVALUATION_FILE = (
    OUTPUT_DIR
    / "evaluation_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not RECONCILIATION_FILE.exists():

        raise FileNotFoundError(
            f"Reconciliation results not found:\n"
            f"{RECONCILIATION_FILE}"
        )

    if not GROUND_TRUTH_FILE.exists():

        raise FileNotFoundError(
            f"Ground truth not found:\n"
            f"{GROUND_TRUTH_FILE}"
        )

    predictions = pd.read_csv(
        RECONCILIATION_FILE
    )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE
    )

    return predictions, ground_truth


# ============================================================
# MERGE PREDICTIONS WITH GROUND TRUTH
# ============================================================

def build_evaluation_table(
    predictions,
    ground_truth
):

    required_prediction_columns = [
        "transaction_id",
        "status",
        "exception_type",
    ]

    required_truth_columns = [
        "transaction_id",
        "true_status",
        "true_exception_type",
    ]

    for column in required_prediction_columns:

        if column not in predictions.columns:

            raise ValueError(
                f"Prediction file missing column: "
                f"{column}"
            )

    for column in required_truth_columns:

        if column not in ground_truth.columns:

            raise ValueError(
                f"Ground truth missing column: "
                f"{column}"
            )

    evaluation = predictions[
        required_prediction_columns
    ].merge(
        ground_truth[
            required_truth_columns
        ],
        on="transaction_id",
        how="inner",
        validate="one_to_one",
    )

    if len(evaluation) != 1000:

        raise ValueError(
            f"Expected 1000 evaluation rows, "
            f"got {len(evaluation)}"
        )

    return evaluation


# ============================================================
# OVERALL STATUS EVALUATION
# ============================================================

def evaluate_status(
    evaluation
):

    print(
        "\n========== OVERALL STATUS EVALUATION =========="
    )

    actual_exception = (
        evaluation["true_status"]
        == "EXCEPTION"
    )

    predicted_exception = (
        evaluation["status"]
        == "EXCEPTION"
    )

    true_positive = (
        actual_exception
        & predicted_exception
    ).sum()

    false_positive = (
        ~actual_exception
        & predicted_exception
    ).sum()

    false_negative = (
        actual_exception
        & ~predicted_exception
    ).sum()

    true_negative = (
        ~actual_exception
        & ~predicted_exception
    ).sum()

    total = len(evaluation)

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive)
        > 0
        else 0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative)
        > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    accuracy = (
        (true_positive + true_negative)
        / total
    )

    print(
        f"True Positives:  {true_positive}"
    )

    print(
        f"False Positives: {false_positive}"
    )

    print(
        f"False Negatives: {false_negative}"
    )

    print(
        f"True Negatives:  {true_negative}"
    )

    print(
        f"\nAccuracy:  {accuracy:.4f}"
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

    return {
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "TN": true_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# EXCEPTION TYPE EVALUATION
# ============================================================

def evaluate_exception_types(
    evaluation
):

    print(
        "\n========== EXCEPTION TYPE EVALUATION =========="
    )

    true_exceptions = evaluation[
        evaluation["true_exception_type"]
        != "NONE"
    ]

    exception_types = sorted(
        true_exceptions[
            "true_exception_type"
        ].unique()
    )

    rows = []

    for exception_type in exception_types:

        actual = (
            evaluation[
                "true_exception_type"
            ]
            == exception_type
        )

        predicted = (
            evaluation[
                "exception_type"
            ]
            == exception_type
        )

        tp = (
            actual & predicted
        ).sum()

        fp = (
            ~actual & predicted
        ).sum()

        fn = (
            actual & ~predicted
        ).sum()

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        actual_count = actual.sum()
        predicted_count = predicted.sum()

        rows.append({

            "exception_type":
                exception_type,

            "actual_count":
                actual_count,

            "predicted_count":
                predicted_count,

            "true_positive":
                tp,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "precision":
                round(
                    precision,
                    4
                ),

            "recall":
                round(
                    recall,
                    4
                ),

            "f1":
                round(
                    f1,
                    4
                ),
        })

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# FIND FALSE POSITIVES
# ============================================================

def show_false_positives(
    evaluation
):

    false_positives = evaluation[
        (
            evaluation["true_status"]
            == "MATCHED"
        )
        &
        (
            evaluation["status"]
            == "EXCEPTION"
        )
    ]

    print(
        "\n========== FALSE POSITIVES =========="
    )

    print(
        f"Count: {len(false_positives)}"
    )

    if not false_positives.empty:

        print(
            false_positives[
                [
                    "transaction_id",
                    "exception_type",
                    "true_exception_type",
                ]
            ]
            .to_string(index=False)
        )

    return false_positives


# ============================================================
# FIND FALSE NEGATIVES
# ============================================================

def show_false_negatives(
    evaluation
):

    false_negatives = evaluation[
        (
            evaluation["true_status"]
            == "EXCEPTION"
        )
        &
        (
            evaluation["status"]
            == "MATCHED"
        )
    ]

    print(
        "\n========== FALSE NEGATIVES =========="
    )

    print(
        f"Count: {len(false_negatives)}"
    )

    if not false_negatives.empty:

        print(
            false_negatives[
                [
                    "transaction_id",
                    "exception_type",
                    "true_exception_type",
                ]
            ]
            .to_string(index=False)
        )

    return false_negatives


# ============================================================
# SAVE EVALUATION
# ============================================================

def save_evaluation(
    evaluation,
    type_results
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation.to_csv(
        EVALUATION_FILE,
        index=False
    )

    type_file = (
        OUTPUT_DIR
        / "exception_type_evaluation.csv"
    )

    type_results.to_csv(
        type_file,
        index=False
    )

    print(
        f"\nDetailed evaluation saved to:"
    )

    print(
        EVALUATION_FILE
    )

    print(
        f"\nException-type evaluation saved to:"
    )

    print(
        type_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "================================================"
    )

    print(
        "        RECONCILIATION EVALUATION"
    )

    print(
        "================================================"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    predictions, ground_truth = (
        load_data()
    )

    print(
        f"\nPredictions: {len(predictions)}"
    )

    print(
        f"Ground truth: {len(ground_truth)}"
    )

    # --------------------------------------------------------
    # Build evaluation table
    # --------------------------------------------------------

    evaluation = (
        build_evaluation_table(
            predictions,
            ground_truth
        )
    )

    # --------------------------------------------------------
    # Overall evaluation
    # --------------------------------------------------------

    evaluate_status(
        evaluation
    )

    # --------------------------------------------------------
    # Exception type evaluation
    # --------------------------------------------------------

    type_results = (
        evaluate_exception_types(
            evaluation
        )
    )

    # --------------------------------------------------------
    # False positives
    # --------------------------------------------------------

    false_positives = (
        show_false_positives(
            evaluation
        )
    )

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    false_negatives = (
        show_false_negatives(
            evaluation
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_evaluation(
        evaluation,
        type_results
    )

    print(
        "\n================================================"
    )

    print(
        "       EVALUATION COMPLETED"
    )

    print(
        "================================================"
    )


if __name__ == "__main__":

    main()