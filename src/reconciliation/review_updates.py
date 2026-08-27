from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"

INPUT_FILE = RECON_DIR / "audit_events.csv"

OUTPUT_FILE = RECON_DIR / "review_queue.csv"


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
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


# ============================================================
# LOAD
# ============================================================

def load_events():

    print("=" * 70)
    print("        PHASE 7.3 — REVIEW / AUDIT UPDATES")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Audit events file not found:\n{INPUT_FILE}"
        )

    events = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Audit event records: {len(events)}"
    )

    return events


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(events):

    print()
    print(
        "========== INPUT VALIDATION =========="
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in events.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    print(
        "[OK] Input schema valid"
    )

    if len(events) != 200:

        raise ValueError(
            f"Expected 200 audit events, "
            f"found {len(events)}"
        )

    print(
        "[OK] 200 audit events loaded"
    )

    if events[
        "event_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate event IDs detected"
        )

    print(
        "[OK] Event IDs unique"
    )

    if events[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected"
        )

    print(
        "[OK] Transaction IDs unique"
    )


# ============================================================
# BUILD REVIEW QUEUE
# ============================================================

def build_review_queue(events):

    print()
    print(
        "========== BUILDING REVIEW QUEUE =========="
    )

    queue = events.copy()

    # --------------------------------------------------------
    # Review case ID
    # --------------------------------------------------------

    queue.insert(
        0,
        "review_case_id",
        [
            f"REVIEW_{i:05d}"
            for i in range(1, len(queue) + 1)
        ],
    )

    # --------------------------------------------------------
    # Review priority
    #
    # Derived from the existing escalation priority.
    # We do not create a new financial-risk calculation.
    # --------------------------------------------------------

    def get_review_priority(row):

        priority = str(
            row["escalation_priority"]
        ).upper()

        if priority == "URGENT":
            return "URGENT"

        if priority == "HIGH":
            return "HIGH"

        if priority == "NORMAL":
            return "NORMAL"

        # Non-escalated low-severity cases.
        if str(row["severity"]).upper() == "LOW":
            return "NORMAL"

        return "HIGH"

    queue[
        "review_priority"
    ] = queue.apply(
        get_review_priority,
        axis=1,
    )

    # --------------------------------------------------------
    # Review queue status
    #
    # No human reviewer has acted yet.
    # --------------------------------------------------------

    queue[
        "queue_status"
    ] = "WAITING_FOR_REVIEW"

    # --------------------------------------------------------
    # Reviewer fields remain empty.
    # --------------------------------------------------------

    queue[
        "reviewer"
    ] = pd.NA

    queue[
        "reviewer_action"
    ] = pd.NA

    queue[
        "reviewer_note"
    ] = pd.NA

    # --------------------------------------------------------
    # Review status remains PENDING.
    # --------------------------------------------------------

    queue[
        "review_status"
    ] = "PENDING"

    # --------------------------------------------------------
    # Review mode
    # --------------------------------------------------------

    queue[
        "review_mode"
    ] = "SIMULATED"

    # --------------------------------------------------------
    # Human review is required for all exception cases
    # represented in this queue.
    # --------------------------------------------------------

    queue[
        "requires_human_review"
    ] = "YES"

    # --------------------------------------------------------
    # Final schema
    # --------------------------------------------------------

    final_columns = [
        "review_case_id",

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

        "review_priority",
        "queue_status",
        "review_status",
        "reviewer",
        "reviewer_action",
        "reviewer_note",

        "requires_human_review",
        "review_mode",
    ]

    queue = queue[
        final_columns
    ]

    print(
        f"Review queue records created: {len(queue)}"
    )

    return queue


# ============================================================
# VALIDATE REVIEW QUEUE
# ============================================================

def validate_review_queue(queue):

    print()
    print("=" * 70)
    print("        PHASE 7.3 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    if len(queue) != 200:

        raise ValueError(
            f"Expected 200 review records, "
            f"found {len(queue)}"
        )

    print(
        "[OK] Review queue count: 200"
    )

    # --------------------------------------------------------
    # Review case IDs
    # --------------------------------------------------------

    if queue[
        "review_case_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate review case IDs detected"
        )

    print(
        "[OK] Review case IDs unique"
    )

    # --------------------------------------------------------
    # Event IDs
    # --------------------------------------------------------

    if queue[
        "event_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate event IDs detected"
        )

    print(
        "[OK] Event IDs unique"
    )

    # --------------------------------------------------------
    # Transaction IDs
    # --------------------------------------------------------

    if queue[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected"
        )

    print(
        "[OK] Transaction IDs unique"
    )

    # --------------------------------------------------------
    # Human review
    # --------------------------------------------------------

    if set(
        queue[
            "requires_human_review"
        ]
    ) != {"YES"}:

        raise ValueError(
            "Human-review requirement invalid"
        )

    print(
        "[OK] Human-review requirement valid"
    )

    # --------------------------------------------------------
    # Review mode
    # --------------------------------------------------------

    if set(
        queue[
            "review_mode"
        ]
    ) != {"SIMULATED"}:

        raise ValueError(
            "Review mode must be SIMULATED"
        )

    print(
        "[OK] Review mode = SIMULATED"
    )

    # --------------------------------------------------------
    # Queue status
    # --------------------------------------------------------

    if set(
        queue[
            "queue_status"
        ]
    ) != {"WAITING_FOR_REVIEW"}:

        raise ValueError(
            "Invalid queue status"
        )

    print(
        "[OK] Queue status = WAITING_FOR_REVIEW"
    )

    # --------------------------------------------------------
    # Review status
    # --------------------------------------------------------

    if set(
        queue[
            "review_status"
        ]
    ) != {"PENDING"}:

        raise ValueError(
            "Review status should remain PENDING"
        )

    print(
        "[OK] Review status = PENDING"
    )

    # --------------------------------------------------------
    # Reviewer fields
    # --------------------------------------------------------

    reviewer_fields = [
        "reviewer",
        "reviewer_action",
        "reviewer_note",
    ]

    populated_reviewer_values = (
        queue[reviewer_fields]
        .notna()
        .sum()
        .sum()
    )

    if populated_reviewer_values != 0:

        raise ValueError(
            "Reviewer fields must remain empty "
            "until a human reviewer acts"
        )

    print(
        "[OK] Reviewer fields correctly pending"
    )

    # --------------------------------------------------------
    # Escalation distribution
    # --------------------------------------------------------

    print()
    print(
        "Review priority distribution:"
    )

    print(
        queue[
            "review_priority"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Queue status distribution:"
    )

    print(
        queue[
            "queue_status"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Review status distribution:"
    )

    print(
        queue[
            "review_status"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Review event distribution:"
    )

    print(
        queue[
            "event_type"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        events = load_events()

        validate_input(
            events
        )

        queue = build_review_queue(
            events
        )

        validate_review_queue(
            queue
        )

        queue.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Review queue saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 7.3 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 7.3 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()