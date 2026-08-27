from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"

AUDIT_FILE = RECON_DIR / "audit_log.csv"
EVENT_FILE = RECON_DIR / "audit_events.csv"
REVIEW_FILE = RECON_DIR / "review_queue.csv"


# ============================================================
# LOAD FILES
# ============================================================

def load_files():

    print("=" * 70)
    print("        PHASE 7.4 — FINAL AUDIT VALIDATION")
    print("=" * 70)

    files = {
        "audit": AUDIT_FILE,
        "events": EVENT_FILE,
        "review": REVIEW_FILE,
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
    print("[OK] All Phase 7 files loaded")

    return data


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_counts(data):

    print()
    print("========== BASIC COUNT VALIDATION ==========")

    expected_counts = {
        "audit": 200,
        "events": 200,
        "review": 200,
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
# ID VALIDATION
# ============================================================

def validate_ids(data):

    print()
    print("========== ID VALIDATION ==========")

    for name, df in data.items():

        if df[
            "transaction_id"
        ].duplicated().any():

            raise ValueError(
                f"{name}: duplicate transaction IDs"
            )

        print(
            f"[OK] {name}: transaction IDs unique"
        )

    # --------------------------------------------------------
    # Audit IDs
    # --------------------------------------------------------

    if data["audit"][
        "audit_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate audit IDs"
        )

    print(
        "[OK] Audit IDs unique"
    )

    # --------------------------------------------------------
    # Event IDs
    # --------------------------------------------------------

    if data["events"][
        "event_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate event IDs"
        )

    print(
        "[OK] Event IDs unique"
    )

    # --------------------------------------------------------
    # Review case IDs
    # --------------------------------------------------------

    if data["review"][
        "review_case_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate review case IDs"
        )

    print(
        "[OK] Review case IDs unique"
    )


# ============================================================
# TRANSACTION FLOW
# ============================================================

def validate_transaction_flow(data):

    print()
    print(
        "========== TRANSACTION FLOW VALIDATION =========="
    )

    audit_ids = set(
        data["audit"]["transaction_id"]
    )

    event_ids = set(
        data["events"]["transaction_id"]
    )

    review_ids = set(
        data["review"]["transaction_id"]
    )

    if audit_ids != event_ids:

        raise ValueError(
            "Audit and event transaction sets differ"
        )

    print(
        "[OK] Audit → event transactions preserved"
    )

    if audit_ids != review_ids:

        raise ValueError(
            "Audit and review transaction sets differ"
        )

    print(
        "[OK] Event → review transactions preserved"
    )

    if len(audit_ids) != 200:

        raise ValueError(
            "Expected 200 unique exception transactions"
        )

    print(
        "[OK] All 200 exception transactions preserved"
    )


# ============================================================
# EXCEPTION CONSISTENCY
# ============================================================

def validate_exception_consistency(data):

    print()
    print(
        "========== EXCEPTION CONSISTENCY VALIDATION =========="
    )

    audit = data["audit"].set_index(
        "transaction_id"
    )

    events = data["events"].set_index(
        "transaction_id"
    )

    review = data["review"].set_index(
        "transaction_id"
    )

    # --------------------------------------------------------
    # Exception type
    # --------------------------------------------------------

    if not (
        audit["exception_type"]
        == events["exception_type"]
    ).all():

        raise ValueError(
            "Exception types changed between audit and events"
        )

    if not (
        audit["exception_type"]
        == review["exception_type"]
    ).all():

        raise ValueError(
            "Exception types changed between audit and review"
        )

    print(
        "[OK] Exception types preserved"
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    if not (
        audit["severity"]
        == events["severity"]
    ).all():

        raise ValueError(
            "Severity changed between audit and events"
        )

    if not (
        audit["severity"]
        == review["severity"]
    ).all():

        raise ValueError(
            "Severity changed between audit and review"
        )

    print(
        "[OK] Severity preserved"
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    if not (
        audit["difference"]
        == events["difference"]
    ).all():

        raise ValueError(
            "Difference changed between audit and events"
        )

    print(
        "[OK] Financial difference preserved"
    )


# ============================================================
# AI / ML CONSISTENCY
# ============================================================

def validate_ai_ml_consistency(data):

    print()
    print(
        "========== AI / ML CONSISTENCY VALIDATION =========="
    )

    audit = data["audit"].set_index(
        "transaction_id"
    )

    events = data["events"].set_index(
        "transaction_id"
    )

    review = data["review"].set_index(
        "transaction_id"
    )

    # --------------------------------------------------------
    # Hybrid decisions
    # --------------------------------------------------------

    if not (
        audit["hybrid_decision"]
        == events["hybrid_decision"]
    ).all():

        raise ValueError(
            "Hybrid decisions changed"
        )

    if not (
        audit["hybrid_decision"]
        == review["hybrid_decision"]
    ).all():

        raise ValueError(
            "Hybrid decisions changed in review queue"
        )

    print(
        "[OK] Hybrid decisions preserved"
    )

    # --------------------------------------------------------
    # Agent decisions
    # --------------------------------------------------------

    if not (
        audit["agent_decision"]
        == events["agent_decision"]
    ).all():

        raise ValueError(
            "Agent decisions changed"
        )

    if not (
        audit["agent_decision"]
        == review["agent_decision"]
    ).all():

        raise ValueError(
            "Agent decisions changed in review queue"
        )

    print(
        "[OK] Agent decisions preserved"
    )


# ============================================================
# ACTION / ESCALATION VALIDATION
# ============================================================

def validate_actions(data):

    print()
    print(
        "========== ACTION / ESCALATION VALIDATION =========="
    )

    audit = data["audit"]
    events = data["events"]
    review = data["review"]

    # --------------------------------------------------------
    # Simulated mode
    # --------------------------------------------------------

    if set(
        audit["execution_mode"]
    ) != {"SIMULATED"}:

        raise ValueError(
            "Audit execution mode is not SIMULATED"
        )

    if set(
        events["execution_mode"]
    ) != {"SIMULATED"}:

        raise ValueError(
            "Event execution mode is not SIMULATED"
        )

    if set(
        review["execution_mode"]
    ) != {"SIMULATED"}:

        raise ValueError(
            "Review execution mode is not SIMULATED"
        )

    print(
        "[OK] All execution modes remain SIMULATED"
    )

    # --------------------------------------------------------
    # Escalation requirement
    # --------------------------------------------------------

    audit_escalations = (
        audit["escalation_required"]
        .astype(str)
        .str.upper()
        == "YES"
    ).sum()

    event_escalations = (
        events["escalation_required"]
        .astype(str)
        .str.upper()
        == "YES"
    ).sum()

    review_escalations = (
        review["escalation_required"]
        .astype(str)
        .str.upper()
        == "YES"
    ).sum()

    if audit_escalations != 170:

        raise ValueError(
            f"Expected 170 audit escalations, "
            f"found {audit_escalations}"
        )

    if event_escalations != 170:

        raise ValueError(
            f"Expected 170 event escalations, "
            f"found {event_escalations}"
        )

    if review_escalations != 170:

        raise ValueError(
            f"Expected 170 review escalations, "
            f"found {review_escalations}"
        )

    print(
        "[OK] 170 escalation requirements preserved"
    )


# ============================================================
# REVIEW STATE VALIDATION
# ============================================================

def validate_review_state(data):

    print()
    print(
        "========== REVIEW STATE VALIDATION =========="
    )

    review = data["review"]

    # --------------------------------------------------------
    # Queue state
    # --------------------------------------------------------

    if set(
        review["queue_status"]
    ) != {"WAITING_FOR_REVIEW"}:

        raise ValueError(
            "Invalid review queue status"
        )

    print(
        "[OK] All review cases waiting for review"
    )

    # --------------------------------------------------------
    # Review state
    # --------------------------------------------------------

    if set(
        review["review_status"]
    ) != {"PENDING"}:

        raise ValueError(
            "Unexpected completed review detected"
        )

    print(
        "[OK] All review cases remain PENDING"
    )

    # --------------------------------------------------------
    # Reviewer fields
    # --------------------------------------------------------

    reviewer_fields = [
        "reviewer",
        "reviewer_action",
        "reviewer_note",
    ]

    populated = (
        review[reviewer_fields]
        .notna()
        .sum()
        .sum()
    )

    if populated != 0:

        raise ValueError(
            "Reviewer fields contain unexpected values"
        )

    print(
        "[OK] No fabricated reviewer activity"
    )


# ============================================================
# EVENT VALIDATION
# ============================================================

def validate_events(data):

    print()
    print(
        "========== EVENT VALIDATION =========="
    )

    events = data["events"]

    # --------------------------------------------------------
    # Event status
    # --------------------------------------------------------

    if set(
        events["event_status"]
    ) != {"OPEN"}:

        raise ValueError(
            "Unexpected event status"
        )

    print(
        "[OK] All audit events remain OPEN"
    )

    # --------------------------------------------------------
    # Event mode
    # --------------------------------------------------------

    if set(
        events["event_mode"]
    ) != {"SIMULATED"}:

        raise ValueError(
            "Invalid event mode"
        )

    print(
        "[OK] Event mode = SIMULATED"
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
    # Event types
    # --------------------------------------------------------

    valid_event_types = {
        "ESCALATION_CREATED",
        "DATE_REVIEW_CREATED",
        "FEE_REVIEW_CREATED",
        "SETTLEMENT_REVIEW_CREATED",
        "RECONCILIATION_EXCEPTION",
    }

    actual_event_types = set(
        events["event_type"]
    )

    if not actual_event_types.issubset(
        valid_event_types
    ):

        raise ValueError(
            "Unknown event type detected"
        )

    print(
        "[OK] Event types valid"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(data):

    audit = data["audit"]
    events = data["events"]
    review = data["review"]

    print()
    print("=" * 70)
    print(
        "        PHASE 7 FINAL SUMMARY"
    )
    print("=" * 70)

    print(
        f"Audit records:        {len(audit)}"
    )

    print(
        f"Audit events:         {len(events)}"
    )

    print(
        f"Review cases:         {len(review)}"
    )

    print()

    print(
        "Exception distribution:"
    )

    print(
        audit["exception_type"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Severity distribution:"
    )

    print(
        audit["severity"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Escalation distribution:"
    )

    print(
        audit["escalation_required"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Review status distribution:"
    )

    print(
        review["review_status"]
        .value_counts()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        data = load_files()

        validate_counts(
            data
        )

        validate_ids(
            data
        )

        validate_transaction_flow(
            data
        )

        validate_exception_consistency(
            data
        )

        validate_ai_ml_consistency(
            data
        )

        validate_actions(
            data
        )

        validate_review_state(
            data
        )

        validate_events(
            data
        )

        print_summary(
            data
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 7.4 COMPLETED"
        )
        print(
            "       PHASE 7 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 7.4 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()