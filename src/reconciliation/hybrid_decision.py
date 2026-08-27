from pathlib import Path
import sys

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RECON_DIR = DATA_DIR / "reconciliation"

RECONCILIATION_FILE = (
    RECON_DIR / "reconciliation_results.csv"
)

ML_DATASET_FILE = (
    RECON_DIR / "ml_dataset.csv"
)

BINARY_MODEL_FILE = (
    RECON_DIR / "binary_model.joblib"
)

MULTICLASS_MODEL_FILE = (
    RECON_DIR / "multiclass_model.joblib"
)

OUTPUT_FILE = (
    RECON_DIR / "hybrid_decisions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# ML is supporting evidence only.
# Rules remain authoritative.

ML_HIGH_CONFIDENCE = 0.80


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("        PHASE 5.6 — HYBRID DECISION LAYER")
    print("=" * 70)

    required_files = [
        RECONCILIATION_FILE,
        ML_DATASET_FILE,
        BINARY_MODEL_FILE,
        MULTICLASS_MODEL_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    reconciliation = pd.read_csv(
        RECONCILIATION_FILE
    )

    ml_dataset = pd.read_csv(
        ML_DATASET_FILE
    )

    binary_pipeline = joblib.load(
        BINARY_MODEL_FILE
    )

    multiclass_pipeline = joblib.load(
        MULTICLASS_MODEL_FILE
    )

    print(
        f"Reconciliation records: "
        f"{len(reconciliation)}"
    )

    print(
        f"ML records: {len(ml_dataset)}"
    )

    print("[OK] Input files loaded")

    return (
        reconciliation,
        ml_dataset,
        binary_pipeline,
        multiclass_pipeline,
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    excluded = {
        "transaction_id",
        "target_exception",
        "target_exception_type",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    return df[feature_columns].copy()


# ============================================================
# GENERATE ML PREDICTIONS
# ============================================================

def generate_predictions(
    ml_dataset,
    binary_pipeline,
    multiclass_pipeline,
):

    print()
    print(
        "Generating ML predictions..."
    )

    X = prepare_features(
        ml_dataset
    )

    # --------------------------------------------------------
    # Binary model
    # --------------------------------------------------------

    binary_preprocessor = (
        binary_pipeline["preprocessor"]
    )

    binary_model = (
        binary_pipeline["model"]
    )

    X_binary = (
        binary_preprocessor.transform(X)
    )

    binary_prediction = (
        binary_model.predict(X_binary)
    )

    binary_probability = (
        binary_model.predict_proba(
            X_binary
        )[:, 1]
    )

    # --------------------------------------------------------
    # Multiclass model
    # --------------------------------------------------------

    multiclass_preprocessor = (
        multiclass_pipeline["preprocessor"]
    )

    multiclass_model = (
        multiclass_pipeline["model"]
    )

    X_multiclass = (
        multiclass_preprocessor.transform(X)
    )

    multiclass_prediction = (
        multiclass_model.predict(
            X_multiclass
        )
    )

    multiclass_probabilities = (
        multiclass_model.predict_proba(
            X_multiclass
        )
    )

    multiclass_classes = (
        multiclass_model.classes_
    )

    multiclass_max_probability = (
        multiclass_probabilities.max(
            axis=1
        )
    )

    return (
        binary_prediction,
        binary_probability,
        multiclass_prediction,
        multiclass_max_probability,
        multiclass_classes,
    )


# ============================================================
# BUILD HYBRID DECISIONS
# ============================================================

def build_decisions(
    reconciliation,
    binary_prediction,
    binary_probability,
    multiclass_prediction,
    multiclass_max_probability,
):

    result = pd.DataFrame(
        {
            "transaction_id":
                reconciliation[
                    "transaction_id"
                ],

            "rule_status":
                reconciliation[
                    "status"
                ],

            "rule_exception_type":
                reconciliation[
                    "exception_type"
                ],

            "ml_exception_prediction":
                binary_prediction,

            "ml_exception_probability":
                binary_probability,

            "ml_predicted_exception_type":
                multiclass_prediction,

            "ml_exception_type_probability":
                multiclass_max_probability,
        }
    )

    # --------------------------------------------------------
    # Convert ML binary prediction
    # --------------------------------------------------------

    result[
        "ml_status"
    ] = result[
        "ml_exception_prediction"
    ].map(
        {
            0: "MATCHED",
            1: "EXCEPTION",
        }
    )

    # --------------------------------------------------------
    # Rule / ML agreement
    # --------------------------------------------------------

    result[
        "ml_agrees_with_rule_status"
    ] = (
        result["rule_status"]
        == result["ml_status"]
    )

    result[
        "ml_agrees_with_rule_exception_type"
    ] = (
        result["rule_exception_type"]
        == result[
            "ml_predicted_exception_type"
        ]
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decisions = []

    for _, row in result.iterrows():

        rule_status = row[
            "rule_status"
        ]

        ml_status = row[
            "ml_status"
        ]

        ml_probability = row[
            "ml_exception_probability"
        ]

        type_agreement = row[
            "ml_agrees_with_rule_exception_type"
        ]

        type_probability = row[
            "ml_exception_type_probability"
        ]

        # ----------------------------------------------------
        # Rule is authoritative
        # ----------------------------------------------------

        if rule_status == "EXCEPTION":

            if (
                ml_status == "EXCEPTION"
                and type_agreement
                and ml_probability
                    >= ML_HIGH_CONFIDENCE
                and type_probability
                    >= ML_HIGH_CONFIDENCE
            ):

                decision = (
                    "RULE_CONFIRMED_BY_ML"
                )

            elif ml_status == "EXCEPTION":

                decision = (
                    "RULE_EXCEPTION_ML_SUPPORT"
                )

            else:

                decision = (
                    "RULE_EXCEPTION_ML_DISAGREEMENT"
                )

        # ----------------------------------------------------
        # Rule says matched
        # ----------------------------------------------------

        else:

            if (
                ml_status == "EXCEPTION"
                and ml_probability
                    >= ML_HIGH_CONFIDENCE
            ):

                decision = (
                    "RULE_MATCHED_ML_REVIEW"
                )

            else:

                decision = (
                    "RULE_MATCHED"
                )

        decisions.append(
            decision
        )

    result[
        "hybrid_decision"
    ] = decisions

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_result(result):

    print()
    print("=" * 70)
    print("        PHASE 5.6 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    if len(result) != 1000:

        raise ValueError(
            "Expected exactly 1000 hybrid records."
        )

    print(
        "[OK] Record count: 1000"
    )

    # --------------------------------------------------------
    # Unique transactions
    # --------------------------------------------------------

    if result[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    print(
        "[OK] Transaction IDs unique"
    )

    # --------------------------------------------------------
    # Rule result preservation
    # --------------------------------------------------------

    if result[
        "rule_status"
    ].isna().any():

        raise ValueError(
            "Missing rule status."
        )

    if result[
        "rule_exception_type"
    ].isna().any():

        raise ValueError(
            "Missing rule exception type."
        )

    print(
        "[OK] Rule results preserved"
    )

    # --------------------------------------------------------
    # ML probabilities
    # --------------------------------------------------------

    if (
        result[
            "ml_exception_probability"
        ]
        .lt(0)
        .any()
        or
        result[
            "ml_exception_probability"
        ]
        .gt(1)
        .any()
    ):

        raise ValueError(
            "Invalid ML exception probabilities."
        )

    print(
        "[OK] ML probabilities valid"
    )

    # --------------------------------------------------------
    # Decision validation
    # --------------------------------------------------------

    valid_decisions = {
        "RULE_MATCHED",
        "RULE_MATCHED_ML_REVIEW",
        "RULE_CONFIRMED_BY_ML",
        "RULE_EXCEPTION_ML_SUPPORT",
        "RULE_EXCEPTION_ML_DISAGREEMENT",
    }

    actual_decisions = set(
        result[
            "hybrid_decision"
        ]
    )

    invalid = (
        actual_decisions
        - valid_decisions
    )

    if invalid:

        raise ValueError(
            f"Invalid hybrid decisions: "
            f"{invalid}"
        )

    print(
        "[OK] Hybrid decisions valid"
    )

    # --------------------------------------------------------
    # Decision distribution
    # --------------------------------------------------------

    print()
    print(
        "Hybrid decision distribution:"
    )

    print(
        result[
            "hybrid_decision"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Rule / ML agreement
    # --------------------------------------------------------

    print()
    print(
        "Rule / ML status agreement:"
    )

    print(
        result[
            "ml_agrees_with_rule_status"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Rule / ML exception-type agreement:"
    )

    print(
        result[
            "ml_agrees_with_rule_exception_type"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # High confidence ML disagreement
    # --------------------------------------------------------

    high_confidence_disagreement = result[
        (
            result[
                "rule_status"
            ] == "EXCEPTION"
        )
        &
        (
            result[
                "ml_status"
            ] == "MATCHED"
        )
        &
        (
            result[
                "ml_exception_probability"
            ] < ML_HIGH_CONFIDENCE
        )
    ]

    print()
    print(
        "Rule EXCEPTION but ML predicts MATCHED:"
    )

    print(
        len(
            high_confidence_disagreement
        )
    )

    # --------------------------------------------------------
    # Rule matched / ML suspicious
    # --------------------------------------------------------

    suspicious_matches = result[
        (
            result[
                "rule_status"
            ] == "MATCHED"
        )
        &
        (
            result[
                "ml_status"
            ] == "EXCEPTION"
        )
        &
        (
            result[
                "ml_exception_probability"
            ] >= ML_HIGH_CONFIDENCE
        )
    ]

    print()
    print(
        "Rule MATCHED but high-confidence ML EXCEPTION:"
    )

    print(
        len(
            suspicious_matches
        )
    )

    if len(suspicious_matches):

        print()
        print(
            suspicious_matches[
                [
                    "transaction_id",
                    "rule_status",
                    "ml_status",
                    "ml_exception_probability",
                    "ml_predicted_exception_type",
                    "ml_exception_type_probability",
                ]
            ].to_string(index=False)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        (
            reconciliation,
            ml_dataset,
            binary_pipeline,
            multiclass_pipeline,
        ) = load_data()

        (
            binary_prediction,
            binary_probability,
            multiclass_prediction,
            multiclass_max_probability,
            multiclass_classes,
        ) = generate_predictions(
            ml_dataset,
            binary_pipeline,
            multiclass_pipeline,
        )

        result = build_decisions(
            reconciliation,
            binary_prediction,
            binary_probability,
            multiclass_prediction,
            multiclass_max_probability,
        )

        validate_result(
            result
        )

        result.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Hybrid decisions saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 5.6 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 5.6 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()