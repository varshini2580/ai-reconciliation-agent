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
DECISION_FILE = RECON_DIR / "agent_decisions.csv"
ACTION_FILE = RECON_DIR / "agent_actions.csv"

OUTPUT_FILE = RECON_DIR / "escalation_cases.csv"


# ============================================================
# ESCALATION PRIORITY
# ============================================================

ESCALATION_PRIORITY = {
    "HIGH": "URGENT",
    "MEDIUM": "HIGH",
    "LOW": "NORMAL",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_inputs():

    print("=" * 70)
    print("        PHASE 6.4 — ESCALATION HANDLING")
    print("=" * 70)

    files = {
        "agent_context": INPUT_FILE,
        "agent_decisions": DECISION_FILE,
        "agent_actions": ACTION_FILE,
    }

    for name, path in files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

    context = pd.read_csv(INPUT_FILE)
    decisions = pd.read_csv(DECISION_FILE)
    actions = pd.read_csv(ACTION_FILE)

    print(
        f"Agent context records: {len(context)}"
    )

    print(
        f"Agent decision records: {len(decisions)}"
    )

    print(
        f"Agent action records: {len(actions)}"
    )

    print("[OK] Escalation inputs loaded")

    return context, decisions, actions


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    context,
    decisions,
    actions,
):

    print()
    print("========== INPUT VALIDATION ==========")

    for name, df in [
        ("context", context),
        ("decisions", decisions),
        ("actions", actions),
    ]:

        if "transaction_id" not in df.columns:

            raise ValueError(
                f"{name} missing transaction_id"
            )

        if df[
            "transaction_id"
        ].duplicated().any():

            raise ValueError(
                f"{name} contains duplicate "
                "transaction IDs"
            )

        print(
            f"[OK] {name} schema valid"
        )


# ============================================================
# BUILD ESCALATION CASES
# ============================================================

def build_escalation_cases(
    context,
    decisions,
    actions,
):

    print()
    print(
        "========== BUILDING ESCALATION CASES =========="
    )

    # --------------------------------------------------------
    # Only cases explicitly requiring escalation are included.
    # --------------------------------------------------------

    escalation = decisions[
        decisions[
            "agent_decision"
        ]
        == "ESCALATE_FOR_REVIEW"
    ].copy()

    print(
        f"Escalation candidates: "
        f"{len(escalation)}"
    )

    # --------------------------------------------------------
    # Select case information.
    # --------------------------------------------------------

    context_fields = [
        "transaction_id",
        "exception_type",
        "severity",
        "difference",
        "explanation",
        "recommended_action",
        "priority",
        "resolution_category",
        "next_step",
        "escalation_required",
        "ml_status",
        "ml_exception_probability",
        "ml_predicted_exception_type",
        "ml_exception_type_probability",
        "hybrid_decision",
    ]

    decision_fields = [
        "transaction_id",
        "agent_decision",
        "agent_decision_reason",
    ]

    action_fields = [
        "transaction_id",
        "action_type",
        "action_status",
        "requires_human",
        "execution_mode",
    ]

    context_part = context[
        context_fields
    ].copy()

    decision_part = decisions[
        decision_fields
    ].copy()

    action_part = actions[
        action_fields
    ].copy()

    # --------------------------------------------------------
    # Merge context + decision.
    # --------------------------------------------------------

    result = escalation[
        ["transaction_id"]
    ].merge(
        context_part,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        decision_part,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        action_part,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Assign escalation priority.
    # --------------------------------------------------------

    result[
        "escalation_priority"
    ] = result[
        "severity"
    ].map(
        ESCALATION_PRIORITY
    )

    # --------------------------------------------------------
    # Generate escalation reason.
    # --------------------------------------------------------

    escalation_reasons = []

    for _, row in result.iterrows():

        hybrid_decision = str(
            row["hybrid_decision"]
        )

        severity = str(
            row["severity"]
        ).upper()

        exception_type = row[
            "exception_type"
        ]

        if (
            hybrid_decision
            == "RULE_EXCEPTION_ML_DISAGREEMENT"
        ):

            reason = (
                f"Human review required because "
                f"the deterministic reconciliation rule "
                f"identified {exception_type}, but the "
                f"ML layer disagreed with the rule result."
            )

        elif severity == "HIGH":

            reason = (
                f"High-severity {exception_type} "
                f"requires human financial review."
            )

        else:

            reason = (
                f"The {exception_type} exception "
                f"requires human review according to "
                f"the reconciliation resolution policy."
            )

        escalation_reasons.append(
            reason
        )

    result[
        "escalation_reason"
    ] = escalation_reasons

    # --------------------------------------------------------
    # Add case status.
    # --------------------------------------------------------

    result[
        "escalation_status"
    ] = "OPEN"

    # --------------------------------------------------------
    # Mark review ownership.
    # --------------------------------------------------------

    result[
        "review_owner"
    ] = "RECONCILIATION_TEAM"

    # --------------------------------------------------------
    # Execution mode remains simulated.
    # --------------------------------------------------------

    result[
        "escalation_mode"
    ] = "SIMULATED"

    print(
        f"Escalation cases created: "
        f"{len(result)}"
    )

    return result


# ============================================================
# VALIDATE ESCALATION CASES
# ============================================================

def validate_escalations(
    result,
    context,
    decisions,
    actions,
):

    print()
    print("=" * 70)
    print("        PHASE 6.4 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    expected_count = (
        decisions[
            "agent_decision"
        ]
        == "ESCALATE_FOR_REVIEW"
    ).sum()

    if len(result) != expected_count:

        raise ValueError(
            f"Expected {expected_count} escalation "
            f"cases, found {len(result)}"
        )

    print(
        f"[OK] Escalation record count: "
        f"{len(result)}"
    )

    # --------------------------------------------------------
    # Expected current count
    # --------------------------------------------------------

    if len(result) != 170:

        raise ValueError(
            f"Expected 170 escalation cases "
            f"for current dataset, found {len(result)}"
        )

    print(
        "[OK] All 170 required escalations captured"
    )

    # --------------------------------------------------------
    # Unique transactions
    # --------------------------------------------------------

    if result[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate escalation transaction IDs"
        )

    print(
        "[OK] Transaction IDs unique"
    )

    # --------------------------------------------------------
    # Escalation requirement
    # --------------------------------------------------------

    invalid_escalation = result[
        result[
            "escalation_required"
        ].astype(str).str.upper()
        != "YES"
    ]

    if len(invalid_escalation):

        raise ValueError(
            "Escalation case contains "
            "escalation_required != YES"
        )

    print(
        "[OK] Escalation requirement preserved"
    )

    # --------------------------------------------------------
    # Agent decision
    # --------------------------------------------------------

    invalid_decisions = result[
        result[
            "agent_decision"
        ]
        != "ESCALATE_FOR_REVIEW"
    ]

    if len(invalid_decisions):

        raise ValueError(
            "Non-escalation decision found "
            "inside escalation cases"
        )

    print(
        "[OK] Agent decisions valid"
    )

    # --------------------------------------------------------
    # Action consistency
    # --------------------------------------------------------

    invalid_actions = result[
        result[
            "action_type"
        ]
        != "CREATE_ESCALATION_CASE"
    ]

    if len(invalid_actions):

        raise ValueError(
            "Invalid escalation action type detected"
        )

    print(
        "[OK] Escalation actions consistent"
    )

    # --------------------------------------------------------
    # Human review
    # --------------------------------------------------------

    invalid_human = result[
        result[
            "requires_human"
        ].astype(str).str.upper()
        != "YES"
    ]

    if len(invalid_human):

        raise ValueError(
            "Escalation case does not require "
            "human review"
        )

    print(
        "[OK] Human review requirement valid"
    )

    # --------------------------------------------------------
    # Escalation priority
    # --------------------------------------------------------

    valid_priorities = {
        "URGENT",
        "HIGH",
        "NORMAL",
    }

    invalid_priorities = (
        set(
            result[
                "escalation_priority"
            ]
        )
        - valid_priorities
    )

    if invalid_priorities:

        raise ValueError(
            f"Invalid escalation priorities: "
            f"{invalid_priorities}"
        )

    print(
        "[OK] Escalation priorities valid"
    )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required_fields = [
        "transaction_id",
        "exception_type",
        "severity",
        "difference",
        "explanation",
        "recommended_action",
        "resolution_category",
        "next_step",
        "hybrid_decision",
        "agent_decision",
        "action_type",
        "escalation_priority",
        "escalation_reason",
        "escalation_status",
        "review_owner",
        "escalation_mode",
    ]

    missing_values = (
        result[required_fields]
        .isna()
        .sum()
        .sum()
    )

    if missing_values:

        raise ValueError(
            f"Missing escalation case values: "
            f"{missing_values}"
        )

    print(
        "[OK] No missing escalation case values"
    )

    # --------------------------------------------------------
    # Escalation status
    # --------------------------------------------------------

    if set(
        result[
            "escalation_status"
        ]
    ) != {"OPEN"}:

        raise ValueError(
            "Invalid escalation status"
        )

    print(
        "[OK] Escalation status valid"
    )

    # --------------------------------------------------------
    # Simulated mode
    # --------------------------------------------------------

    if set(
        result[
            "escalation_mode"
        ]
    ) != {"SIMULATED"}:

        raise ValueError(
            "Escalation mode must be SIMULATED"
        )

    print(
        "[OK] Escalation mode is SIMULATED"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print()
    print(
        "Escalation priority distribution:"
    )

    print(
        result[
            "escalation_priority"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Escalation exception distribution:"
    )

    print(
        result[
            "exception_type"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Escalation reason distribution:"
    )

    print(
        result[
            "hybrid_decision"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # ML disagreement cases
    # --------------------------------------------------------

    disagreements = result[
        result[
            "hybrid_decision"
        ]
        == "RULE_EXCEPTION_ML_DISAGREEMENT"
    ]

    print()
    print(
        "ML disagreement escalations:"
    )

    print(
        len(disagreements)
    )

    if len(disagreements):

        print()
        print(
            disagreements[
                [
                    "transaction_id",
                    "exception_type",
                    "severity",
                    "ml_predicted_exception_type",
                    "ml_exception_probability",
                    "hybrid_decision",
                    "escalation_priority",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print()
    print(
        "========== SAMPLE ESCALATION CASES =========="
    )

    print(
        result[
            [
                "transaction_id",
                "exception_type",
                "severity",
                "escalation_priority",
                "hybrid_decision",
                "agent_decision",
                "escalation_status",
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

        (
            context,
            decisions,
            actions,
        ) = load_inputs()

        validate_inputs(
            context,
            decisions,
            actions,
        )

        result = build_escalation_cases(
            context,
            decisions,
            actions,
        )

        validate_escalations(
            result,
            context,
            decisions,
            actions,
        )

        result.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Escalation cases saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 6.4 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 6.4 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()