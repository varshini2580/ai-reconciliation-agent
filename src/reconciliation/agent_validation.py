from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"

RECONCILIATION_FILE = RECON_DIR / "reconciliation_results.csv"
EXPLANATION_FILE = RECON_DIR / "exception_explanations.csv"
RESOLUTION_FILE = RECON_DIR / "resolution_actions.csv"
HYBRID_FILE = RECON_DIR / "hybrid_decisions.csv"
CONTEXT_FILE = RECON_DIR / "agent_context.csv"
DECISION_FILE = RECON_DIR / "agent_decisions.csv"
ACTION_FILE = RECON_DIR / "agent_actions.csv"
ESCALATION_FILE = RECON_DIR / "escalation_cases.csv"


# ============================================================
# LOAD
# ============================================================

def load_files():

    print("=" * 70)
    print("        PHASE 6.5 — END-TO-END AGENT VALIDATION")
    print("=" * 70)

    files = {
        "reconciliation": RECONCILIATION_FILE,
        "explanation": EXPLANATION_FILE,
        "resolution": RESOLUTION_FILE,
        "hybrid": HYBRID_FILE,
        "context": CONTEXT_FILE,
        "decision": DECISION_FILE,
        "action": ACTION_FILE,
        "escalation": ESCALATION_FILE,
    }

    for name, path in files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

    data = {
        name: pd.read_csv(path)
        for name, path in files.items()
    }

    print()
    for name, df in data.items():
        print(
            f"{name.capitalize()} records: {len(df)}"
        )

    print()
    print("[OK] All Phase 6 files loaded")

    return data


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_basic_counts(data):

    print()
    print("========== BASIC COUNT VALIDATION ==========")

    expected_counts = {
        "reconciliation": 1000,
        "explanation": 1000,
        "resolution": 200,
        "hybrid": 1000,
        "context": 200,
        "decision": 200,
        "action": 200,
        "escalation": 170,
    }

    for name, expected in expected_counts.items():

        actual = len(data[name])

        if actual != expected:

            raise ValueError(
                f"{name}: expected {expected}, "
                f"found {actual}"
            )

        print(
            f"[OK] {name}: {actual} records"
        )


# ============================================================
# UNIQUE ID VALIDATION
# ============================================================

def validate_unique_ids(data):

    print()
    print(
        "========== TRANSACTION ID VALIDATION =========="
    )

    for name, df in data.items():

        if df[
            "transaction_id"
        ].duplicated().any():

            raise ValueError(
                f"{name} contains duplicate "
                "transaction IDs"
            )

        print(
            f"[OK] {name}: transaction IDs unique"
        )


# ============================================================
# FULL TRANSACTION PRESERVATION
# ============================================================

def validate_full_transaction_flow(data):

    print()
    print(
        "========== TRANSACTION FLOW VALIDATION =========="
    )

    base_ids = set(
        data[
            "reconciliation"
        ]["transaction_id"]
    )

    for name in [
        "explanation",
        "hybrid",
    ]:

        ids = set(
            data[name]["transaction_id"]
        )

        if ids != base_ids:

            raise ValueError(
                f"{name} does not preserve "
                "all 1000 transaction IDs"
            )

        print(
            f"[OK] {name}: all 1000 transactions preserved"
        )


# ============================================================
# EXCEPTION FLOW
# ============================================================

def validate_exception_flow(data):

    print()
    print(
        "========== EXCEPTION FLOW VALIDATION =========="
    )

    reconciliation = data[
        "reconciliation"
    ]

    expected_exception_ids = set(
        reconciliation[
            reconciliation["status"]
            == "EXCEPTION"
        ]["transaction_id"]
    )

    print(
        f"Expected exception cases: "
        f"{len(expected_exception_ids)}"
    )

    for name in [
        "resolution",
        "context",
        "decision",
        "action",
    ]:

        actual_ids = set(
            data[name]["transaction_id"]
        )

        if actual_ids != expected_exception_ids:

            missing = (
                expected_exception_ids
                - actual_ids
            )

            extra = (
                actual_ids
                - expected_exception_ids
            )

            raise ValueError(
                f"{name} exception flow mismatch. "
                f"Missing: {missing}, Extra: {extra}"
            )

        print(
            f"[OK] {name}: all 200 exception cases preserved"
        )


# ============================================================
# ESCALATION FLOW
# ============================================================

def validate_escalation_flow(data):

    print()
    print(
        "========== ESCALATION FLOW VALIDATION =========="
    )

    decisions = data[
        "decision"
    ]

    expected_escalations = set(
        decisions[
            decisions[
                "agent_decision"
            ]
            == "ESCALATE_FOR_REVIEW"
        ]["transaction_id"]
    )

    actual_escalations = set(
        data[
            "escalation"
        ]["transaction_id"]
    )

    if expected_escalations != actual_escalations:

        missing = (
            expected_escalations
            - actual_escalations
        )

        extra = (
            actual_escalations
            - expected_escalations
        )

        raise ValueError(
            f"Escalation flow mismatch. "
            f"Missing: {missing}, Extra: {extra}"
        )

    print(
        f"[OK] All {len(expected_escalations)} "
        "escalation decisions captured"
    )

    # --------------------------------------------------------
    # Every escalation must require human review.
    # --------------------------------------------------------

    escalation = data[
        "escalation"
    ]

    invalid_human = escalation[
        escalation[
            "requires_human"
        ].astype(str).str.upper()
        != "YES"
    ]

    if len(invalid_human):

        raise ValueError(
            "Escalation cases without human review found"
        )

    print(
        "[OK] All escalation cases require human review"
    )


# ============================================================
# RULE / ML CONSISTENCY
# ============================================================

def validate_hybrid_consistency(data):

    print()
    print(
        "========== HYBRID CONSISTENCY VALIDATION =========="
    )

    hybrid = data[
        "hybrid"
    ]

    # --------------------------------------------------------
    # Rule status must be preserved.
    # --------------------------------------------------------

    reconciliation = data[
        "reconciliation"
    ][
        [
            "transaction_id",
            "status",
            "exception_type",
        ]
    ]

    merged = reconciliation.merge(
        hybrid[
            [
                "transaction_id",
                "rule_status",
                "rule_exception_type",
            ]
        ],
        on="transaction_id",
        how="inner",
        validate="one_to_one",
    )

    if not (
        merged["status"]
        == merged["rule_status"]
    ).all():

        raise ValueError(
            "Hybrid layer changed rule status"
        )

    if not (
        merged["exception_type"]
        == merged["rule_exception_type"]
    ).all():

        raise ValueError(
            "Hybrid layer changed rule exception type"
        )

    print(
        "[OK] Hybrid layer preserves rule results"
    )

    # --------------------------------------------------------
    # ML disagreement cases are review signals,
    # not rule overrides.
    # --------------------------------------------------------

    disagreement = hybrid[
        hybrid[
            "hybrid_decision"
        ].isin(
            [
                "RULE_EXCEPTION_ML_DISAGREEMENT",
                "RULE_MATCHED_ML_REVIEW",
            ]
        )
    ]

    print(
        f"[OK] ML review signals identified: "
        f"{len(disagreement)}"
    )

    print(
        "[OK] ML does not override deterministic rules"
    )


# ============================================================
# AGENT DECISION CONSISTENCY
# ============================================================

def validate_agent_decisions(data):

    print()
    print(
        "========== AGENT DECISION VALIDATION =========="
    )

    decisions = data[
        "decision"
    ]

    # --------------------------------------------------------
    # Escalation requirement must lead to escalation.
    # --------------------------------------------------------

    escalation_required = decisions[
        decisions[
            "escalation_required"
        ].astype(str).str.upper()
        == "YES"
    ]

    invalid = escalation_required[
        escalation_required[
            "agent_decision"
        ]
        != "ESCALATE_FOR_REVIEW"
    ]

    if len(invalid):

        raise ValueError(
            "Escalation-required cases were not escalated"
        )

    print(
        "[OK] Escalation requirements respected"
    )

    # --------------------------------------------------------
    # Non-escalated cases must not be marked escalated.
    # --------------------------------------------------------

    no_escalation = decisions[
        decisions[
            "escalation_required"
        ].astype(str).str.upper()
        == "NO"
    ]

    invalid = no_escalation[
        no_escalation[
            "agent_decision"
        ]
        == "ESCALATE_FOR_REVIEW"
    ]

    if len(invalid):

        raise ValueError(
            "Non-escalated cases were incorrectly escalated"
        )

    print(
        "[OK] Non-escalated decisions consistent"
    )


# ============================================================
# ACTION CONSISTENCY
# ============================================================

def validate_actions(data):

    print()
    print(
        "========== ACTION FLOW VALIDATION =========="
    )

    decisions = data[
        "decision"
    ]

    actions = data[
        "action"
    ]

    merged = decisions[
        [
            "transaction_id",
            "agent_decision",
        ]
    ].merge(
        actions[
            [
                "transaction_id",
                "agent_decision",
                "execution_mode",
                "requires_human",
            ]
        ],
        on="transaction_id",
        how="inner",
        suffixes=(
            "_decision",
            "_action",
        ),
        validate="one_to_one",
    )

    if not (
        merged[
            "agent_decision_decision"
        ]
        == merged[
            "agent_decision_action"
        ]
    ).all():

        raise ValueError(
            "Action layer changed agent decisions"
        )

    print(
        "[OK] Action layer preserves agent decisions"
    )

    if set(
        merged[
            "execution_mode"
        ]
    ) != {"SIMULATED"}:

        raise ValueError(
            "Non-simulated action detected"
        )

    print(
        "[OK] All actions remain simulated"
    )

    if not (
        merged[
            "requires_human"
        ].astype(str).str.upper()
        == "YES"
    ).all():

        raise ValueError(
            "Action layer lost human-review requirement"
        )

    print(
        "[OK] Human-review requirement preserved"
    )


# ============================================================
# END-TO-END SUMMARY
# ============================================================

def print_summary(data):

    reconciliation = data["reconciliation"]
    decisions = data["decision"]
    actions = data["action"]
    escalation = data["escalation"]

    print()
    print("=" * 70)
    print("        END-TO-END AGENT SUMMARY")
    print("=" * 70)

    total_transactions = len(reconciliation)

    matched_transactions = (
        reconciliation["status"] == "MATCHED"
    ).sum()

    exception_transactions = (
        reconciliation["status"] == "EXCEPTION"
    ).sum()

    print(
        f"Total transactions: {total_transactions}"
    )

    print(
        f"Matched transactions: {matched_transactions}"
    )

    print(
        f"Exception transactions: {exception_transactions}"
    )

    print(
        f"Agent decisions: {len(decisions)}"
    )

    print(
        f"Agent actions: {len(actions)}"
    )

    print(
        f"Escalation cases: {len(escalation)}"
    )

    print()
    print(
        "Agent decision distribution:"
    )

    print(
        decisions["agent_decision"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Action status distribution:"
    )

    print(
        actions["action_status"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Escalation priority distribution:"
    )

    print(
        escalation["escalation_priority"]
        .value_counts()
        .to_string()
    )

# ============================================================
# MAIN
# ============================================================

def main():

    try:

        data = load_files()

        validate_basic_counts(
            data
        )

        validate_unique_ids(
            data
        )

        validate_full_transaction_flow(
            data
        )

        validate_exception_flow(
            data
        )

        validate_escalation_flow(
            data
        )

        validate_hybrid_consistency(
            data
        )

        validate_agent_decisions(
            data
        )

        validate_actions(
            data
        )

        print_summary(
            data
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 6.5 COMPLETED"
        )
        print(
            "       PHASE 6 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 6.5 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()