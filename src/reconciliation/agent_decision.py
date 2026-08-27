from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"

INPUT_FILE = RECON_DIR / "agent_context.csv"

OUTPUT_FILE = RECON_DIR / "agent_decisions.csv"


# ============================================================
# DECISION MAPPING
# ============================================================

DECISION_BY_CATEGORY = {
    "FINANCIAL_REVIEW": "FINANCIAL_REVIEW",
    "SETTLEMENT_REVIEW": "SETTLEMENT_REVIEW",
    "PAYMENT_REVIEW": "PAYMENT_REVIEW",
    "DUPLICATE_REVIEW": "DUPLICATE_REVIEW",
    "REFUND_REVIEW": "REFUND_REVIEW",
    "DISPUTE_REVIEW": "DISPUTE_REVIEW",
    "FEE_REVIEW": "FEE_REVIEW",
    "REFERENCE_REVIEW": "REFERENCE_REVIEW",
    "DATE_REVIEW": "DATE_REVIEW",
    "ADJUSTMENT_REVIEW": "ADJUSTMENT_REVIEW",
}


VALID_DECISIONS = {
    "ESCALATE_FOR_REVIEW",
    "FINANCIAL_REVIEW",
    "SETTLEMENT_REVIEW",
    "PAYMENT_REVIEW",
    "DUPLICATE_REVIEW",
    "REFUND_REVIEW",
    "DISPUTE_REVIEW",
    "FEE_REVIEW",
    "REFERENCE_REVIEW",
    "DATE_REVIEW",
    "ADJUSTMENT_REVIEW",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_context():

    print("=" * 70)
    print("        PHASE 6.2 — AGENT DECISION WORKFLOW")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Agent context file not found:\n{INPUT_FILE}"
        )

    context = pd.read_csv(INPUT_FILE)

    print(
        f"Agent context records: {len(context)}"
    )

    print("[OK] Agent context loaded")

    return context


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(context):

    print()
    print("========== INPUT VALIDATION ==========")

    required_columns = [
        "transaction_id",
        "exception_type",
        "severity",
        "priority",
        "resolution_category",
        "next_step",
        "escalation_required",
        "hybrid_decision",
    ]

    missing = [
        column
        for column in required_columns
        if column not in context.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if len(context) != 200:

        raise ValueError(
            f"Expected 200 exception records, "
            f"found {len(context)}"
        )

    if context[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs found."
        )

    print(
        "[OK] Input schema valid"
    )

    print(
        "[OK] 200 exception records loaded"
    )

    print(
        "[OK] Transaction IDs unique"
    )


# ============================================================
# BUILD AGENT DECISIONS
# ============================================================

def build_decisions(context):

    print()
    print("========== BUILDING AGENT DECISIONS ==========")

    decisions = []

    decision_reasons = []

    for _, row in context.iterrows():

        category = row[
            "resolution_category"
        ]

        escalation = str(
            row["escalation_required"]
        ).upper()

        severity = str(
            row["severity"]
        ).upper()

        hybrid_decision = str(
            row["hybrid_decision"]
        )

        # ----------------------------------------------------
        # Escalation has highest priority.
        #
        # The agent does NOT automatically resolve a case
        # that the resolution layer has marked for escalation.
        # ----------------------------------------------------

        if escalation == "YES":

            decision = "ESCALATE_FOR_REVIEW"

            if (
                hybrid_decision
                == "RULE_EXCEPTION_ML_DISAGREEMENT"
            ):

                reason = (
                    "The reconciliation rule identified an "
                    "exception requiring escalation, while "
                    "the ML layer disagreed with the rule result."
                )

            elif severity == "HIGH":

                reason = (
                    "The exception is high priority and "
                    "requires human review before resolution."
                )

            else:

                reason = (
                    "The resolution workflow requires this "
                    "exception to be escalated for review."
                )

        # ----------------------------------------------------
        # Non-escalated cases use their resolution category.
        # ----------------------------------------------------

        else:

            if category not in DECISION_BY_CATEGORY:

                raise ValueError(
                    f"Unknown resolution category: "
                    f"{category}"
                )

            decision = (
                DECISION_BY_CATEGORY[
                    category
                ]
            )

            reason = (
                f"The exception is categorized as "
                f"{category} and does not require escalation."
            )

        decisions.append(
            decision
        )

        decision_reasons.append(
            reason
        )

    result = context.copy()

    result[
        "agent_decision"
    ] = decisions

    result[
        "agent_decision_reason"
    ] = decision_reasons

    return result


# ============================================================
# VALIDATE DECISIONS
# ============================================================

def validate_decisions(result):

    print()
    print("=" * 70)
    print("        PHASE 6.2 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    if len(result) != 200:

        raise ValueError(
            f"Expected 200 decisions, "
            f"found {len(result)}"
        )

    print(
        "[OK] Decision record count valid"
    )

    # --------------------------------------------------------
    # Transaction IDs
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
    # Decision validity
    # --------------------------------------------------------

    actual_decisions = set(
        result[
            "agent_decision"
        ]
    )

    invalid = (
        actual_decisions
        - VALID_DECISIONS
    )

    if invalid:

        raise ValueError(
            f"Invalid agent decisions: {invalid}"
        )

    print(
        "[OK] All agent decisions valid"
    )

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    if result[
        "agent_decision_reason"
    ].isna().any():

        raise ValueError(
            "Missing decision reasons."
        )

    print(
        "[OK] Decision reasons present"
    )

    # --------------------------------------------------------
    # Escalation consistency
    # --------------------------------------------------------

    escalated = result[
        result[
            "escalation_required"
        ].astype(str).str.upper()
        == "YES"
    ]

    invalid_escalation = escalated[
        escalated[
            "agent_decision"
        ]
        != "ESCALATE_FOR_REVIEW"
    ]

    if len(invalid_escalation):

        raise ValueError(
            "Some escalation-required cases "
            "were not assigned ESCALATE_FOR_REVIEW."
        )

    print(
        "[OK] Escalation decisions consistent"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print()
    print(
        "Agent decision distribution:"
    )

    print(
        result[
            "agent_decision"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Priority distribution
    # --------------------------------------------------------

    print()
    print(
        "Priority distribution:"
    )

    print(
        result[
            "priority"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Escalation distribution
    # --------------------------------------------------------

    print()
    print(
        "Escalation distribution:"
    )

    print(
        result[
            "escalation_required"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print()
    print(
        "========== SAMPLE AGENT DECISIONS =========="
    )

    print(
        result[
            [
                "transaction_id",
                "exception_type",
                "severity",
                "resolution_category",
                "escalation_required",
                "hybrid_decision",
                "agent_decision",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        context = load_context()

        validate_input(
            context
        )

        result = build_decisions(
            context
        )

        validate_decisions(
            result
        )

        result.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Agent decisions saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 6.2 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 6.2 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()