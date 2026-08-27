from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"

INPUT_FILE = RECON_DIR / "agent_decisions.csv"

OUTPUT_FILE = RECON_DIR / "agent_actions.csv"


# ============================================================
# ACTION MAPPING
# ============================================================

ACTION_BY_DECISION = {
    "ESCALATE_FOR_REVIEW": {
        "action_type": "CREATE_ESCALATION_CASE",
        "action_status": "PENDING_REVIEW",
        "requires_human": "YES",
    },

    "FINANCIAL_REVIEW": {
        "action_type": "CREATE_FINANCIAL_REVIEW",
        "action_status": "REVIEW_REQUIRED",
        "requires_human": "YES",
    },

    "SETTLEMENT_REVIEW": {
        "action_type": "CREATE_SETTLEMENT_REVIEW",
        "action_status": "REVIEW_REQUIRED",
        "requires_human": "YES",
    },

    "PAYMENT_REVIEW": {
        "action_type": "CREATE_PAYMENT_REVIEW",
        "action_status": "REVIEW_REQUIRED",
        "requires_human": "YES",
    },

    "DUPLICATE_REVIEW": {
        "action_type": "CREATE_DUPLICATE_REVIEW",
        "action_status": "REVIEW_REQUIRED",
        "requires_human": "YES",
    },

    "REFUND_REVIEW": {
        "action_type": "CREATE_REFUND_REVIEW",
        "action_status": "REVIEW_REQUIRED",
        "requires_human": "YES",
    },

    "DISPUTE_REVIEW": {
        "action_type": "CREATE_DISPUTE_REVIEW",
        "action_status": "REVIEW_REQUIRED",
        "requires_human": "YES",
    },

    "FEE_REVIEW": {
        "action_type": "CREATE_FEE_REVIEW",
        "action_status": "REVIEW_REQUIRED",
        "requires_human": "YES",
    },

    "REFERENCE_REVIEW": {
        "action_type": "CREATE_REFERENCE_REVIEW",
        "action_status": "REVIEW_REQUIRED",
        "requires_human": "YES",
    },

    "DATE_REVIEW": {
        "action_type": "CREATE_DATE_REVIEW",
        "action_status": "REVIEW_REQUIRED",
        "requires_human": "YES",
    },

    "ADJUSTMENT_REVIEW": {
        "action_type": "CREATE_ADJUSTMENT_REVIEW",
        "action_status": "REVIEW_REQUIRED",
        "requires_human": "YES",
    },
}


# ============================================================
# LOAD DECISIONS
# ============================================================

def load_decisions():

    print("=" * 70)
    print("        PHASE 6.3 — AGENT ACTION EXECUTION")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Agent decisions file not found:\n{INPUT_FILE}"
        )

    decisions = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Agent decision records: {len(decisions)}"
    )

    print("[OK] Agent decisions loaded")

    return decisions


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(decisions):

    print()
    print("========== INPUT VALIDATION ==========")

    required_columns = [
        "transaction_id",
        "agent_decision",
        "exception_type",
        "severity",
        "priority",
        "next_step",
        "agent_decision_reason",
    ]

    missing = [
        column
        for column in required_columns
        if column not in decisions.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if len(decisions) != 200:

        raise ValueError(
            f"Expected 200 decision records, "
            f"found {len(decisions)}"
        )

    if decisions[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs found."
        )

    invalid_decisions = (
        set(decisions["agent_decision"])
        - set(ACTION_BY_DECISION)
    )

    if invalid_decisions:

        raise ValueError(
            f"Unknown agent decisions: "
            f"{invalid_decisions}"
        )

    print("[OK] Input schema valid")
    print("[OK] 200 decision records loaded")
    print("[OK] Transaction IDs unique")
    print("[OK] Agent decisions valid")


# ============================================================
# BUILD ACTIONS
# ============================================================

def build_actions(decisions):

    print()
    print("========== BUILDING AGENT ACTIONS ==========")

    actions = []

    for _, row in decisions.iterrows():

        decision = row[
            "agent_decision"
        ]

        action_config = (
            ACTION_BY_DECISION[
                decision
            ]
        )

        # ----------------------------------------------------
        # This is a simulated action.
        #
        # No real financial/payment operation is performed.
        # ----------------------------------------------------

        action_description = (
            f"Simulated action for transaction "
            f"{row['transaction_id']}: "
            f"{decision}. "
            f"Next step: {row['next_step']}"
        )

        actions.append(
            {
                "transaction_id":
                    row["transaction_id"],

                "exception_type":
                    row["exception_type"],

                "severity":
                    row["severity"],

                "priority":
                    row["priority"],

                "agent_decision":
                    decision,

                "action_type":
                    action_config[
                        "action_type"
                    ],

                "action_status":
                    action_config[
                        "action_status"
                    ],

                "requires_human":
                    action_config[
                        "requires_human"
                    ],

                "action_description":
                    action_description,

                "next_step":
                    row["next_step"],

                "hybrid_decision":
                    row["hybrid_decision"],

                "execution_mode":
                    "SIMULATED",
            }
        )

    result = pd.DataFrame(
        actions
    )

    print(
        f"Action records created: {len(result)}"
    )

    return result


# ============================================================
# VALIDATE ACTIONS
# ============================================================

def validate_actions(actions):

    print()
    print("=" * 70)
    print("        PHASE 6.3 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    if len(actions) != 200:

        raise ValueError(
            f"Expected 200 action records, "
            f"found {len(actions)}"
        )

    print(
        "[OK] Action record count valid"
    )

    # --------------------------------------------------------
    # Unique transactions
    # --------------------------------------------------------

    if actions[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    print(
        "[OK] Transaction IDs unique"
    )

    # --------------------------------------------------------
    # Action types
    # --------------------------------------------------------

    if actions[
        "action_type"
    ].isna().any():

        raise ValueError(
            "Missing action types detected."
        )

    print(
        "[OK] Action types present"
    )

    # --------------------------------------------------------
    # Action status
    # --------------------------------------------------------

    valid_statuses = {
        "PENDING_REVIEW",
        "REVIEW_REQUIRED",
    }

    actual_statuses = set(
        actions[
            "action_status"
        ]
    )

    invalid_statuses = (
        actual_statuses
        - valid_statuses
    )

    if invalid_statuses:

        raise ValueError(
            f"Invalid action statuses: "
            f"{invalid_statuses}"
        )

    print(
        "[OK] Action statuses valid"
    )

    # --------------------------------------------------------
    # Human review
    # --------------------------------------------------------

    invalid_human_values = set(
        actions[
            "requires_human"
        ]
    ) - {"YES"}

    if invalid_human_values:

        raise ValueError(
            "All current actions must require "
            "human review."
        )

    print(
        "[OK] Human-review requirement valid"
    )

    # --------------------------------------------------------
    # Execution mode
    # --------------------------------------------------------

    if set(
        actions["execution_mode"]
    ) != {"SIMULATED"}:

        raise ValueError(
            "Unexpected execution mode detected."
        )

    print(
        "[OK] Execution mode is SIMULATED"
    )

    # --------------------------------------------------------
    # Missing critical fields
    # --------------------------------------------------------

    critical_fields = [
        "transaction_id",
        "agent_decision",
        "action_type",
        "action_status",
        "requires_human",
        "action_description",
        "next_step",
    ]

    missing_values = (
        actions[critical_fields]
        .isna()
        .sum()
        .sum()
    )

    if missing_values:

        raise ValueError(
            f"Missing critical action values: "
            f"{missing_values}"
        )

    print(
        "[OK] No missing critical action values"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print()
    print(
        "Action type distribution:"
    )

    print(
        actions[
            "action_type"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Action status distribution:"
    )

    print(
        actions[
            "action_status"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Human review distribution:"
    )

    print(
        actions[
            "requires_human"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print()
    print(
        "========== SAMPLE AGENT ACTIONS =========="
    )

    print(
        actions[
            [
                "transaction_id",
                "exception_type",
                "agent_decision",
                "action_type",
                "action_status",
                "requires_human",
                "execution_mode",
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

        decisions = load_decisions()

        validate_input(
            decisions
        )

        actions = build_actions(
            decisions
        )

        validate_actions(
            actions
        )

        actions.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Agent actions saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 6.3 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 6.3 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()