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
EXPLANATION_FILE = RECON_DIR / "ai_explanations.csv"
HYBRID_FILE = RECON_DIR / "hybrid_decisions.csv"
DECISION_FILE = RECON_DIR / "agent_decisions.csv"
ACTION_FILE = RECON_DIR / "agent_actions.csv"
ESCALATION_FILE = RECON_DIR / "escalation_cases.csv"

OUTPUT_FILE = RECON_DIR / "audit_log.csv"


# ============================================================
# AUDIT SCHEMA
# ============================================================

AUDIT_COLUMNS = [
    "audit_id",
    "transaction_id",
    "exception_type",
    "severity",
    "difference",

    "deterministic_explanation",
    "ai_explanation",
    "explanation_source",
    "recommended_action",

    "ml_status",
    "ml_exception_probability",
    "ml_predicted_exception_type",
    "ml_exception_type_probability",
    "hybrid_decision",

    "agent_decision",
    "agent_decision_reason",

    "action_type",
    "action_status",
    "execution_mode",

    "escalation_required",
    "escalation_priority",
    "escalation_status",

    "review_owner",
    "review_status",
    "reviewer",
    "reviewer_action",
    "reviewer_note",

    "audit_event_status",
]


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():

    print("=" * 70)
    print("        PHASE 7.1 — AUDIT EVENT SCHEMA")
    print("=" * 70)

    files = {
        "reconciliation": RECONCILIATION_FILE,
        "explanation": EXPLANATION_FILE,
        "hybrid": HYBRID_FILE,
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
    print("[OK] All Phase 7 inputs loaded")

    return data


# ============================================================
# VALIDATE SOURCE FILES
# ============================================================

def validate_sources(data):

    print()
    print("========== SOURCE VALIDATION ==========")

    for name, df in data.items():

        if "transaction_id" not in df.columns:

            raise ValueError(
                f"{name} is missing transaction_id"
            )

        if df["transaction_id"].duplicated().any():

            raise ValueError(
                f"{name} contains duplicate transaction IDs"
            )

        print(
            f"[OK] {name} schema valid"
        )


# ============================================================
# BUILD AUDIT LOG
# ============================================================

def build_audit_log(data):

    print()
    print("========== BUILDING AUDIT EVENTS ==========")

    reconciliation = data["reconciliation"]
    explanation = data["explanation"]
    hybrid = data["hybrid"]
    decision = data["decision"]
    action = data["action"]
    escalation = data["escalation"]

    # --------------------------------------------------------
    # Start with actual exception cases.
    # --------------------------------------------------------

    base = reconciliation[
        reconciliation["status"] == "EXCEPTION"
    ][
        [
            "transaction_id",
            "exception_type",
            "difference",
        ]
    ].copy()

    print(
        f"Exception transactions for audit: {len(base)}"
    )

    # --------------------------------------------------------
    # Phase 4B explanation information
    # --------------------------------------------------------

    explanation_fields = [
        "transaction_id",
        "severity",
        "deterministic_explanation",
        "ai_explanation",
        "explanation_source",
        "recommended_action",
    ]

    result = base.merge(
        explanation[explanation_fields],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Hybrid ML information
    # --------------------------------------------------------

    hybrid_fields = [
        "transaction_id",
        "ml_status",
        "ml_exception_probability",
        "ml_predicted_exception_type",
        "ml_exception_type_probability",
        "hybrid_decision",
    ]

    result = result.merge(
        hybrid[hybrid_fields],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Agent decision
    # --------------------------------------------------------

    decision_fields = [
        "transaction_id",
        "agent_decision",
        "agent_decision_reason",
        "escalation_required",
    ]

    result = result.merge(
        decision[decision_fields],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Agent action
    # --------------------------------------------------------

    action_fields = [
        "transaction_id",
        "action_type",
        "action_status",
        "execution_mode",
    ]

    result = result.merge(
        action[action_fields],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Escalation
    #
    # Only 170 of the 200 exception cases are escalated.
    # Therefore this is intentionally a LEFT JOIN.
    # --------------------------------------------------------

    escalation_fields = [
        "transaction_id",
        "escalation_priority",
        "escalation_status",
        "review_owner",
    ]

    result = result.merge(
        escalation[escalation_fields],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Human review state
    #
    # No real reviewer action has occurred yet.
    # Therefore these fields remain empty.
    # --------------------------------------------------------

    result["review_status"] = "PENDING"

    result["reviewer"] = pd.NA

    result["reviewer_action"] = pd.NA

    result["reviewer_note"] = pd.NA

    # --------------------------------------------------------
    # Audit ID
    # --------------------------------------------------------

    result.insert(
        0,
        "audit_id",
        [
            f"AUDIT_{i:05d}"
            for i in range(1, len(result) + 1)
        ],
    )

    # --------------------------------------------------------
    # Current audit state
    # --------------------------------------------------------

    result["audit_event_status"] = "OPEN"

    # --------------------------------------------------------
    # Final audit schema
    # --------------------------------------------------------

    result = result[AUDIT_COLUMNS]

    print(
        f"Audit records created: {len(result)}"
    )

    return result


# ============================================================
# VALIDATE AUDIT LOG
# ============================================================

def validate_audit_log(audit):

    print()
    print("=" * 70)
    print("        PHASE 7.1 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    if len(audit) != 200:

        raise ValueError(
            f"Expected 200 audit records, "
            f"found {len(audit)}"
        )

    print(
        "[OK] Audit record count: 200"
    )

    # --------------------------------------------------------
    # Exact schema
    # --------------------------------------------------------

    actual_columns = list(
        audit.columns
    )

    if actual_columns != AUDIT_COLUMNS:

        raise ValueError(
            "Audit schema does not match "
            "the defined schema."
        )

    print(
        "[OK] Audit schema valid"
    )

    # --------------------------------------------------------
    # Audit IDs
    # --------------------------------------------------------

    if audit["audit_id"].duplicated().any():

        raise ValueError(
            "Duplicate audit IDs detected."
        )

    print(
        "[OK] Audit IDs unique"
    )

    # --------------------------------------------------------
    # Transaction IDs
    # --------------------------------------------------------

    if audit["transaction_id"].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    print(
        "[OK] Transaction IDs unique"
    )

    # --------------------------------------------------------
    # Required populated fields
    # --------------------------------------------------------

    required_fields = [
        "audit_id",
        "transaction_id",
        "exception_type",
        "severity",
        "difference",
        "deterministic_explanation",
        "ai_explanation",
        "recommended_action",
        "ml_status",
        "ml_exception_probability",
        "ml_predicted_exception_type",
        "hybrid_decision",
        "agent_decision",
        "action_type",
        "action_status",
        "execution_mode",
        "escalation_required",
        "audit_event_status",
    ]

    missing = (
        audit[required_fields]
        .isna()
        .sum()
        .sum()
    )

    if missing:

        raise ValueError(
            f"Missing required audit values: {missing}"
        )

    print(
        "[OK] Required audit fields populated"
    )

    # --------------------------------------------------------
    # Reviewer fields
    #
    # These MUST be empty at Phase 7.1 because no real
    # human review has happened yet.
    # --------------------------------------------------------

    reviewer_fields = [
        "reviewer",
        "reviewer_action",
        "reviewer_note",
    ]

    reviewer_values = (
        audit[reviewer_fields]
        .notna()
        .sum()
        .sum()
    )

    if reviewer_values != 0:

        raise ValueError(
            "Reviewer fields should be empty "
            "before human review."
        )

    print(
        "[OK] Reviewer fields correctly pending"
    )

    # --------------------------------------------------------
    # Review status
    # --------------------------------------------------------

    if set(
        audit["review_status"]
    ) != {"PENDING"}:

        raise ValueError(
            "Invalid initial review status."
        )

    print(
        "[OK] Review status = PENDING"
    )

    # --------------------------------------------------------
    # Audit status
    # --------------------------------------------------------

    if set(
        audit["audit_event_status"]
    ) != {"OPEN"}:

        raise ValueError(
            "Invalid initial audit event status."
        )

    print(
        "[OK] Audit event status = OPEN"
    )

    # --------------------------------------------------------
    # Execution mode
    # --------------------------------------------------------

    if set(
        audit["execution_mode"]
    ) != {"SIMULATED"}:

        raise ValueError(
            "Audit contains non-simulated execution."
        )

    print(
        "[OK] Execution mode = SIMULATED"
    )

    # --------------------------------------------------------
    # Escalation consistency
    # --------------------------------------------------------

    escalation_required = audit[
        audit["escalation_required"]
        .astype(str)
        .str.upper()
        == "YES"
    ]

    if len(escalation_required) != 170:

        raise ValueError(
            f"Expected 170 escalation records, "
            f"found {len(escalation_required)}"
        )

    print(
        "[OK] 170 escalation requirements preserved"
    )

    # --------------------------------------------------------
    # ML disagreement count
    # --------------------------------------------------------

    disagreements = audit[
        audit["hybrid_decision"]
        == "RULE_EXCEPTION_ML_DISAGREEMENT"
    ]

    print(
        f"[OK] ML disagreement signals preserved: "
        f"{len(disagreements)}"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print()
    print(
        "Audit exception distribution:"
    )

    print(
        audit["exception_type"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Audit event status:"
    )

    print(
        audit["audit_event_status"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Review status:"
    )

    print(
        audit["review_status"]
        .value_counts()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        data = load_inputs()

        validate_sources(
            data
        )

        audit = build_audit_log(
            data
        )

        validate_audit_log(
            audit
        )

        audit.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Audit log saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 7.1 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 7.1 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()