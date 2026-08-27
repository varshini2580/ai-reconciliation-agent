import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RECONCILIATION_DIR = DATA_DIR / "reconciliation"

INPUT_FILE = RECONCILIATION_DIR / "ai_explanations.csv"
OUTPUT_FILE = RECONCILIATION_DIR / "resolution_actions.csv"


# ============================================================
# RESOLUTION RULES
# ============================================================

RESOLUTION_RULES = {

    "AMOUNT_MISMATCH": {
        "resolution_category": "FINANCIAL_REVIEW",
        "priority": "HIGH",
        "next_step": (
            "Compare the expected transaction amount with "
            "the actual settlement amount and identify the cause "
            "of the variance."
        ),
        "escalation_required": "YES",
    },

    "CHARGEBACK": {
        "resolution_category": "DISPUTE_REVIEW",
        "priority": "HIGH",
        "next_step": (
            "Review the chargeback record, dispute reason, "
            "and supporting transaction information."
        ),
        "escalation_required": "YES",
    },

    "DATE_MISMATCH": {
        "resolution_category": "DATE_REVIEW",
        "priority": "LOW",
        "next_step": (
            "Compare the relevant payment and settlement dates "
            "and verify the correct processing date."
        ),
        "escalation_required": "NO",
    },

    "DUPLICATE_PAYMENT": {
        "resolution_category": "DUPLICATE_REVIEW",
        "priority": "MEDIUM",
        "next_step": (
            "Review the duplicate payment records and determine "
            "whether a reversal or refund is required."
        ),
        "escalation_required": "YES",
    },

    "DUPLICATE_SETTLEMENT": {
        "resolution_category": "DUPLICATE_REVIEW",
        "priority": "MEDIUM",
        "next_step": (
            "Review the settlement records and determine whether "
            "the transaction was settled more than once."
        ),
        "escalation_required": "YES",
    },

    "FAILED_PAYMENT": {
        "resolution_category": "PAYMENT_REVIEW",
        "priority": "HIGH",
        "next_step": (
            "Verify the payment failure status and determine "
            "whether the payment should be retried or followed up."
        ),
        "escalation_required": "YES",
    },

    "INCORRECT_FEE": {
        "resolution_category": "FEE_REVIEW",
        "priority": "MEDIUM",
        "next_step": (
            "Compare the recorded fee with the expected fee "
            "according to the applicable fee structure."
        ),
        "escalation_required": "NO",
    },

    "MISSING_PAYMENT": {
        "resolution_category": "PAYMENT_REVIEW",
        "priority": "HIGH",
        "next_step": (
            "Verify the payment gateway records and confirm "
            "whether the payment was received."
        ),
        "escalation_required": "YES",
    },

    "MISSING_SETTLEMENT": {
        "resolution_category": "SETTLEMENT_REVIEW",
        "priority": "HIGH",
        "next_step": (
            "Verify the settlement batch and check whether "
            "the transaction is still pending."
        ),
        "escalation_required": "YES",
    },

    "MULTIPLE_PAYMENTS": {
        "resolution_category": "DUPLICATE_REVIEW",
        "priority": "MEDIUM",
        "next_step": (
            "Review all payment records associated with the "
            "transaction and determine whether multiple payments "
            "were legitimately received."
        ),
        "escalation_required": "YES",
    },

    "PARTIAL_SETTLEMENT": {
        "resolution_category": "SETTLEMENT_REVIEW",
        "priority": "MEDIUM",
        "next_step": (
            "Verify whether the remaining settlement amount "
            "is pending or was incorrectly omitted."
        ),
        "escalation_required": "YES",
    },

    "REFUND": {
        "resolution_category": "REFUND_REVIEW",
        "priority": "HIGH",
        "next_step": (
            "Verify the refund authorization and confirm that "
            "the refund amount and transaction reference are correct."
        ),
        "escalation_required": "YES",
    },

    "SETTLEMENT_DELAY": {
        "resolution_category": "SETTLEMENT_REVIEW",
        "priority": "LOW",
        "next_step": (
            "Check the settlement processing status and "
            "investigate the reason for the delay."
        ),
        "escalation_required": "NO",
    },

    "UNKNOWN_ADJUSTMENT": {
        "resolution_category": "ADJUSTMENT_REVIEW",
        "priority": "MEDIUM",
        "next_step": (
            "Review the adjustment entry and identify its "
            "source and business reason."
        ),
        "escalation_required": "YES",
    },

    "WRONG_TRANSACTION_REFERENCE": {
        "resolution_category": "REFERENCE_REVIEW",
        "priority": "MEDIUM",
        "next_step": (
            "Verify the transaction reference across the "
            "payment, bank, and settlement records."
        ),
        "escalation_required": "YES",
    },
}


# ============================================================
# LOAD INPUT
# ============================================================

def load_input():

    print("=" * 60)
    print("        LOADING PHASE 5 DATA")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"AI explanation records: {len(df)}")

    required_columns = [
        "transaction_id",
        "exception_type",
        "severity",
        "ai_explanation",
        "recommended_action",
        "explanation_source",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("[OK] Input schema valid")

    return df


# ============================================================
# BUILD RESOLUTION RECORD
# ============================================================

def build_resolution_record(row):

    exception_type = row["exception_type"]

    if exception_type not in RESOLUTION_RULES:
        raise ValueError(
            f"No resolution rule defined for "
            f"{exception_type}"
        )

    rule = RESOLUTION_RULES[exception_type]

    return {
        "transaction_id": row["transaction_id"],
        "exception_type": exception_type,
        "severity": row["severity"],
        "resolution_category": rule[
            "resolution_category"
        ],
        "priority": rule["priority"],
        "next_step": rule["next_step"],
        "escalation_required": rule[
            "escalation_required"
        ],
        "recommended_action": row[
            "recommended_action"
        ],
        "ai_explanation": row[
            "ai_explanation"
        ],
        "explanation_source": row[
            "explanation_source"
        ],
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_results(result_df, input_df):

    print()
    print("=" * 60)
    print("        PHASE 5 VALIDATION")
    print("=" * 60)

    expected_count = len(input_df)

    print(f"Input exception records: {expected_count}")
    print(f"Resolution records: {len(result_df)}")

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    if len(result_df) != expected_count:
        raise ValueError(
            "Resolution record count does not match input."
        )

    # --------------------------------------------------------
    # Transaction uniqueness
    # --------------------------------------------------------

    duplicate_count = (
        result_df["transaction_id"]
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Duplicate transaction IDs found: "
            f"{duplicate_count}"
        )

    # --------------------------------------------------------
    # Exception types
    # --------------------------------------------------------

    input_types = set(
        input_df["exception_type"]
    )

    output_types = set(
        result_df["exception_type"]
    )

    if input_types != output_types:
        raise ValueError(
            "Exception types changed between input "
            "and resolution output."
        )

    # --------------------------------------------------------
    # Required values
    # --------------------------------------------------------

    required_output_columns = [
        "resolution_category",
        "priority",
        "next_step",
        "escalation_required",
    ]

    for column in required_output_columns:

        if result_df[column].isna().any():

            raise ValueError(
                f"Missing values found in {column}."
            )

        if (
            result_df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):

            raise ValueError(
                f"Empty values found in {column}."
            )

    # --------------------------------------------------------
    # Priority validation
    # --------------------------------------------------------

    valid_priorities = {
        "HIGH",
        "MEDIUM",
        "LOW",
    }

    invalid_priorities = set(
        result_df["priority"]
    ) - valid_priorities

    if invalid_priorities:
        raise ValueError(
            f"Invalid priorities: {invalid_priorities}"
        )

    # --------------------------------------------------------
    # Escalation validation
    # --------------------------------------------------------

    valid_escalation_values = {
        "YES",
        "NO",
    }

    invalid_escalation = set(
        result_df["escalation_required"]
    ) - valid_escalation_values

    if invalid_escalation:
        raise ValueError(
            "Invalid escalation values: "
            f"{invalid_escalation}"
        )

    # --------------------------------------------------------
    # Validation summary
    # --------------------------------------------------------

    print("[OK] Record count valid")
    print("[OK] Transaction IDs unique")
    print("[OK] Exception types preserved")
    print("[OK] Resolution categories present")
    print("[OK] Priorities valid")
    print("[OK] Next steps present")
    print("[OK] Escalation values valid")

    print()
    print("Priority distribution:")

    print(
        result_df["priority"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Resolution category distribution:")

    print(
        result_df["resolution_category"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Escalation distribution:")

    print(
        result_df["escalation_required"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        input_df = load_input()

        # ----------------------------------------------------
        # BUILD RESOLUTIONS
        # ----------------------------------------------------

        print()
        print(
            "Building deterministic resolution actions..."
        )

        results = []

        for _, row in input_df.iterrows():

            results.append(
                build_resolution_record(row)
            )

        result_df = pd.DataFrame(results)

        print(
            f"Resolution records created: "
            f"{len(result_df)}"
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        validate_results(
            result_df,
            input_df,
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        result_df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Resolution actions saved to:"
        )

        print(OUTPUT_FILE)

        # ----------------------------------------------------
        # SAMPLE
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("        SAMPLE RESOLUTION ACTIONS")
        print("=" * 60)

        print(
            result_df[
                [
                    "transaction_id",
                    "exception_type",
                    "priority",
                    "resolution_category",
                    "next_step",
                    "escalation_required",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

        print()
        print("=" * 60)
        print("       PHASE 5 COMPLETED")
        print("=" * 60)

    except Exception as exc:

        print()
        print("=" * 60)
        print("       PHASE 5 FAILED")
        print("=" * 60)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()