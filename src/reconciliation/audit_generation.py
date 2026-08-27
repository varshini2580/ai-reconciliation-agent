from pathlib import Path
import sys
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"

INPUT_FILE = RECON_DIR / "audit_log.csv"

OUTPUT_FILE = RECON_DIR / "audit_events.csv"


# ============================================================
# EXPECTED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
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
# EVENT TYPES
# ============================================================

def determine_event_type(row):

    if row["escalation_required"] == "YES":
        return "ESCALATION_CREATED"

    if row["action_type"] == "CREATE_DATE_REVIEW":
        return "DATE_REVIEW_CREATED"

    if row["action_type"] == "CREATE_FEE_REVIEW":
        return "FEE_REVIEW_CREATED"

    if row["action_type"] == "CREATE_SETTLEMENT_REVIEW":
        return "SETTLEMENT_REVIEW_CREATED"

    return "RECONCILIATION_EXCEPTION"


# ============================================================
# LOAD
# ============================================================

def load_audit_log():

    print("=" * 70)
    print("        PHASE 7.2 — AUDIT LOG GENERATION")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Audit log not found:\n{INPUT_FILE}"
        )

    audit = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Audit records: {len(audit)}"
    )

    return audit


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(audit):

    print()
    print(
        "========== INPUT VALIDATION =========="
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in audit.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing audit columns: "
            + ", ".join(missing_columns)
        )

    print(
        "[OK] Audit input schema valid"
    )

    if len(audit) != 200:

        raise ValueError(
            f"Expected 200 audit records, "
            f"found {len(audit)}"
        )

    print(
        "[OK] 200 audit records loaded"
    )

    if audit[
        "audit_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate audit IDs detected"
        )

    print(
        "[OK] Audit IDs unique"
    )

    if audit[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected"
        )

    print(
        "[OK] Transaction IDs unique"
    )


# ============================================================
# BUILD AUDIT EVENTS
# ============================================================

def build_audit_events(audit):

    print()
    print(
        "========== BUILDING AUDIT EVENTS =========="
    )

    events = audit.copy()

    # --------------------------------------------------------
    # Event ID
    # --------------------------------------------------------

    events.insert(
        0,
        "event_id",
        [
            f"EVENT_{i:05d}"
            for i in range(1, len(events) + 1)
        ],
    )

    # --------------------------------------------------------
    # Event timestamp
    #
    # This represents the time at which this audit-generation
    # process recorded the event.
    # --------------------------------------------------------

    event_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    events[
        "event_timestamp"
    ] = event_timestamp

    # --------------------------------------------------------
    # Event type
    # --------------------------------------------------------

    events[
        "event_type"
    ] = events.apply(
        determine_event_type,
        axis=1,
    )

    # --------------------------------------------------------
    # Event actor
    #
    # This is a system-generated audit event.
    # --------------------------------------------------------

    events[
        "event_actor"
    ] = "RECONCILIATION_AGENT"

    # --------------------------------------------------------
    # Event source
    # --------------------------------------------------------

    events[
        "event_source"
    ] = "PHASE_7_AUDIT"

    # --------------------------------------------------------
    # Event mode
    #
    # Keep this simulated.
    # --------------------------------------------------------

    events[
        "event_mode"
    ] = "SIMULATED"

    # --------------------------------------------------------
    # Preserve existing audit state.
    # --------------------------------------------------------

    events[
        "event_status"
    ] = events[
        "audit_event_status"
    ]

    # --------------------------------------------------------
    # Human review remains pending.
    # --------------------------------------------------------

    events[
        "review_status"
    ] = events[
        "review_status"
    ].fillna("PENDING")

    # --------------------------------------------------------
    # Final event schema
    # --------------------------------------------------------

    final_columns = [
        "event_id",
        "audit_id",
        "transaction_id",

        "event_timestamp",
        "event_type",
        "event_actor",
        "event_source",
        "event_mode",
        "event_status",

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
    ]

    events = events[
        final_columns
    ]

    print(
        f"Audit events generated: {len(events)}"
    )

    return events


# ============================================================
# VALIDATE EVENTS
# ============================================================

def validate_events(events):

    print()
    print("=" * 70)
    print("        PHASE 7.2 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    if len(events) != 200:

        raise ValueError(
            f"Expected 200 audit events, "
            f"found {len(events)}"
        )

    print(
        "[OK] Audit event count: 200"
    )

    # --------------------------------------------------------
    # Event IDs
    # --------------------------------------------------------

    if events[
        "event_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate event IDs detected"
        )

    print(
        "[OK] Event IDs unique"
    )

    # --------------------------------------------------------
    # Audit IDs
    # --------------------------------------------------------

    if events[
        "audit_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate audit IDs detected"
        )

    print(
        "[OK] Audit IDs unique"
    )

    # --------------------------------------------------------
    # Transaction IDs
    # --------------------------------------------------------

    if events[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected"
        )

    print(
        "[OK] Transaction IDs unique"
    )

    # --------------------------------------------------------
    # Event timestamp
    # --------------------------------------------------------

    if events[
        "event_timestamp"
    ].isna().any():

        raise ValueError(
            "Missing event timestamps detected"
        )

    print(
        "[OK] Event timestamps present"
    )

    # --------------------------------------------------------
    # Event actor
    # --------------------------------------------------------

    if set(
        events["event_actor"]
    ) != {"RECONCILIATION_AGENT"}:

        raise ValueError(
            "Invalid event actor"
        )

    print(
        "[OK] Event actor valid"
    )

    # --------------------------------------------------------
    # Event source
    # --------------------------------------------------------

    if set(
        events["event_source"]
    ) != {"PHASE_7_AUDIT"}:

        raise ValueError(
            "Invalid event source"
        )

    print(
        "[OK] Event source valid"
    )

    # --------------------------------------------------------
    # Event mode
    # --------------------------------------------------------

    if set(
        events["event_mode"]
    ) != {"SIMULATED"}:

        raise ValueError(
            "Audit event mode must be SIMULATED"
        )

    print(
        "[OK] Event mode = SIMULATED"
    )

    # --------------------------------------------------------
    # Human review
    # --------------------------------------------------------

    if not (
        events[
            "review_status"
        ] == "PENDING"
    ).all():

        raise ValueError(
            "Unexpected completed human review"
        )

    print(
        "[OK] Human review remains PENDING"
    )

    # --------------------------------------------------------
    # Escalation count
    # --------------------------------------------------------

    escalation_count = (
        events[
            "escalation_required"
        ]
        .astype(str)
        .str.upper()
        == "YES"
    ).sum()

    if escalation_count != 170:

        raise ValueError(
            f"Expected 170 escalation events, "
            f"found {escalation_count}"
        )

    print(
        "[OK] 170 escalation events preserved"
    )

    # --------------------------------------------------------
    # Event status
    # --------------------------------------------------------

    if not (
        events[
            "event_status"
        ] == "OPEN"
    ).all():

        raise ValueError(
            "Unexpected event status detected"
        )

    print(
        "[OK] Event status = OPEN"
    )

    # --------------------------------------------------------
    # Event type distribution
    # --------------------------------------------------------

    print()
    print(
        "Event type distribution:"
    )

    print(
        events[
            "event_type"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Severity distribution
    # --------------------------------------------------------

    print()
    print(
        "Severity distribution:"
    )

    print(
        events[
            "severity"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        audit = load_audit_log()

        validate_input(
            audit
        )

        events = build_audit_events(
            audit
        )

        validate_events(
            events
        )

        events.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Audit events saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 7.2 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 7.2 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()