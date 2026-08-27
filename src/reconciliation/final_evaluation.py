from pathlib import Path
import sys
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"


FILES = {
    "reconciliation": RECON_DIR / "reconciliation_results.csv",
    "evaluation": RECON_DIR / "evaluation_results.csv",
    "exception_evaluation": RECON_DIR / "exception_type_evaluation.csv",

    "explanations": RECON_DIR / "ai_explanations.csv",
    "resolution": RECON_DIR / "resolution_actions.csv",

    "ml_dataset": RECON_DIR / "ml_dataset.csv",
    "binary_train": RECON_DIR / "ml_binary_train.csv",
    "binary_test": RECON_DIR / "ml_binary_test.csv",
    "multiclass_train": RECON_DIR / "ml_multiclass_train.csv",
    "multiclass_test": RECON_DIR / "ml_multiclass_test.csv",

    "ml_binary_evaluation": RECON_DIR / "ml_binary_evaluation.csv",
    "ml_multiclass_evaluation": RECON_DIR / "ml_multiclass_evaluation.csv",

    "hybrid": RECON_DIR / "hybrid_decisions.csv",

    "context": RECON_DIR / "agent_context.csv",
    "decisions": RECON_DIR / "agent_decisions.csv",
    "actions": RECON_DIR / "agent_actions.csv",
    "escalations": RECON_DIR / "escalation_cases.csv",

    "audit": RECON_DIR / "audit_log.csv",
    "events": RECON_DIR / "audit_events.csv",
    "review": RECON_DIR / "review_queue.csv",

    "dashboard": RECON_DIR / "dashboard_data.csv",
    "kpis": RECON_DIR / "dashboard_kpis.csv",
    "exception_analytics": RECON_DIR / "exception_analytics.csv",
    "ml_agent_analytics": RECON_DIR / "ml_agent_analytics.csv",
}


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED = {
    "total": 1000,
    "matched": 800,
    "exceptions": 200,

    "explanations": 200,
    "resolution": 200,

    "high": 110,
    "medium": 70,
    "low": 20,

    "ml_confirmed": 120,
    "ml_supported": 71,
    "ml_disagreement": 9,

    "agent_escalation": 170,
    "escalations": 170,

    "audit": 200,
    "events": 200,
    "review": 200,
}


# ============================================================
# LOAD FILES
# ============================================================

def load_files():

    print("=" * 70)
    print("        PHASE 8.1 — FINAL END-TO-END EVALUATION")
    print("=" * 70)

    data = {}

    for name, path in FILES.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        data[name] = pd.read_csv(path)

        print(
            f"{name:24s}: {len(data[name])} records"
        )

    print()
    print("[OK] All Phase 8 input files loaded")

    return data


# ============================================================
# BASIC FILE COUNTS
# ============================================================

def validate_counts(data):

    print()
    print("========== FILE COUNT VALIDATION ==========")

    expected = {
        "reconciliation": 1000,
        "evaluation": 1000,
        "explanations": 200,
        "resolution": 200,

        "ml_dataset": 1000,
        "binary_train": 800,
        "binary_test": 200,
        "multiclass_train": 800,
        "multiclass_test": 200,

        "hybrid": 1000,

        "context": 200,
        "decisions": 200,
        "actions": 200,
        "escalations": 170,

        "audit": 200,
        "events": 200,
        "review": 200,

        "dashboard": 1000,
        "kpis": 45,
        "exception_analytics": 38,
        "ml_agent_analytics": 23,
    }

    for name, expected_count in expected.items():

        actual = len(data[name])

        if actual != expected_count:

            raise ValueError(
                f"{name}: expected {expected_count}, "
                f"found {actual}"
            )

        print(
            f"[OK] {name}: {actual}"
        )


# ============================================================
# TRANSACTION ID VALIDATION
# ============================================================

def validate_transaction_ids(data):

    print()
    print("========== TRANSACTION ID VALIDATION ==========")

    files = [
        "reconciliation",
        "evaluation",
        "explanations",
        "resolution",
        "ml_dataset",
        "hybrid",
        "context",
        "decisions",
        "actions",
        "escalations",
        "audit",
        "events",
        "review",
        "dashboard",
    ]

    for name in files:

        df = data[name]

        if "transaction_id" not in df.columns:

            raise ValueError(
                f"{name}: transaction_id missing"
            )

        if df[
            "transaction_id"
        ].duplicated().any():

            raise ValueError(
                f"{name}: duplicate transaction IDs"
            )

        print(
            f"[OK] {name}: transaction IDs unique"
        )


# ============================================================
# RECONCILIATION VALIDATION
# ============================================================

def validate_reconciliation(data):

    print()
    print("========== RECONCILIATION VALIDATION ==========")

    df = data["reconciliation"]

    if "status" not in df.columns:

        raise ValueError(
            "reconciliation_results.csv has no status column"
        )

    matched = (
        df["status"] == "MATCHED"
    ).sum()

    exceptions = (
        df["status"] == "EXCEPTION"
    ).sum()

    if matched != 800:

        raise ValueError(
            f"Expected 800 matched, found {matched}"
        )

    if exceptions != 200:

        raise ValueError(
            f"Expected 200 exceptions, found {exceptions}"
        )

    print(
        "[OK] Matched transactions: 800"
    )

    print(
        "[OK] Exception transactions: 200"
    )

    rate = matched / len(df) * 100

    if round(rate, 2) != 80.0:

        raise ValueError(
            f"Expected reconciliation rate 80%, found {rate}%"
        )

    print(
        "[OK] Reconciliation rate: 80.00%"
    )


# ============================================================
# EVALUATION FILE VALIDATION
# ============================================================

def validate_evaluation(data):

    print()
    print("========== PHASE 3 EVALUATION VALIDATION ==========")

    df = data["evaluation"]

    if len(df) != 1000:

        raise ValueError(
            "Evaluation results must contain 1000 records"
        )

    print(
        "[OK] Evaluation results: 1000 records"
    )

    if "transaction_id" in df.columns:

        if df[
            "transaction_id"
        ].duplicated().any():

            raise ValueError(
                "Evaluation transaction IDs are duplicated"
            )

        print(
            "[OK] Evaluation transaction IDs unique"
        )

    print(
        "[OK] Phase 3 evaluation output preserved"
    )


# ============================================================
# EXCEPTION TYPE VALIDATION
# ============================================================

def validate_exceptions(data):

    print()
    print("========== EXCEPTION VALIDATION ==========")

    df = data["reconciliation"]

    exceptions = df[
        df["status"] == "EXCEPTION"
    ]

    if len(exceptions) != 200:

        raise ValueError(
            "Exception count mismatch"
        )

    expected_types = {
        "AMOUNT_MISMATCH": 30,
        "CHARGEBACK": 10,
        "DATE_MISMATCH": 10,
        "DUPLICATE_PAYMENT": 15,
        "DUPLICATE_SETTLEMENT": 10,
        "FAILED_PAYMENT": 15,
        "INCORRECT_FEE": 10,
        "MISSING_PAYMENT": 20,
        "MISSING_SETTLEMENT": 20,
        "MULTIPLE_PAYMENTS": 5,
        "PARTIAL_SETTLEMENT": 15,
        "REFUND": 15,
        "SETTLEMENT_DELAY": 10,
        "UNKNOWN_ADJUSTMENT": 5,
        "WRONG_TRANSACTION_REFERENCE": 10,
    }

    actual = (
        exceptions[
            "exception_type"
        ]
        .value_counts()
        .to_dict()
    )

    for exception_type, expected_count in expected_types.items():

        found = actual.get(
            exception_type,
            0,
        )

        if found != expected_count:

            raise ValueError(
                f"{exception_type}: expected "
                f"{expected_count}, found {found}"
            )

    print(
        "[OK] All 15 exception types preserved"
    )

    print(
        "[OK] Exception distribution matches expected ground truth"
    )


# ============================================================
# SEVERITY VALIDATION
# ============================================================

def validate_severity(data):

    print()
    print("========== SEVERITY VALIDATION ==========")

    # Severity is introduced in the exception/AI layer,
    # not in reconciliation_results.csv.

    df = data["explanations"]

    if "severity" not in df.columns:

        raise ValueError(
            "Severity column is missing from ai_explanations.csv"
        )

    counts = (
        df["severity"]
        .value_counts()
        .to_dict()
    )

    expected = {
        "HIGH": 110,
        "MEDIUM": 70,
        "LOW": 20,
    }

    total = 0

    for severity, expected_count in expected.items():

        actual = counts.get(
            severity,
            0,
        )

        if actual != expected_count:

            raise ValueError(
                f"{severity}: expected "
                f"{expected_count}, found {actual}"
            )

        total += actual

        print(
            f"[OK] {severity}: {actual}"
        )

    if total != 200:

        raise ValueError(
            f"Expected 200 severity records, found {total}"
        )

    print(
        "[OK] Severity distribution: 110 HIGH / 70 MEDIUM / 20 LOW"
    )


# ============================================================
# AI EXPLANATION VALIDATION
# ============================================================

def validate_ai(data):

    print()
    print("========== AI EXPLANATION VALIDATION ==========")

    df = data["explanations"]

    if len(df) != 200:

        raise ValueError(
            "Expected 200 AI explanations"
        )

    if "ai_explanation" not in df.columns:

        raise ValueError(
            "ai_explanation column missing"
        )

    missing = df[
        "ai_explanation"
    ].isna().sum()

    if missing != 0:

        raise ValueError(
            f"{missing} AI explanations missing"
        )

    print(
        "[OK] 200 AI explanations"
    )

    print(
        "[OK] No missing AI explanations"
    )


# ============================================================
# RESOLUTION VALIDATION
# ============================================================

def validate_resolution(data):

    print()
    print("========== RESOLUTION VALIDATION ==========")

    df = data["resolution"]

    if len(df) != 200:

        raise ValueError(
            "Expected 200 resolution records"
        )

    print(
        "[OK] 200 resolution records"
    )

    required = [
        "transaction_id",
        "exception_type",
        "priority",
        "resolution_category",
        "next_step",
        "escalation_required",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing resolution columns: {missing}"
        )

    print(
        "[OK] Resolution fields present"
    )


# ============================================================
# ML DATASET VALIDATION
# ============================================================

def validate_ml_dataset(data):

    print()
    print("========== ML DATASET VALIDATION ==========")

    df = data["ml_dataset"]

    if len(df) != 1000:

        raise ValueError(
            "ML dataset must contain 1000 records"
        )

    print(
        "[OK] ML dataset: 1000"
    )

    if "target_exception" in df.columns:

        exception_count = (
            df["target_exception"] == 1
        ).sum()

        if exception_count != 200:

            raise ValueError(
                f"ML target expected 200 exceptions, "
                f"found {exception_count}"
            )

        print(
            "[OK] ML target: 200 exceptions"
        )

    print(
        "[OK] ML feature dataset preserved"
    )


# ============================================================
# ML TEST SPLITS
# ============================================================

def validate_ml_splits(data):

    print()
    print("========== ML SPLIT VALIDATION ==========")

    if len(data["binary_train"]) != 800:
        raise ValueError(
            "Binary train set must contain 800 records"
        )

    if len(data["binary_test"]) != 200:
        raise ValueError(
            "Binary test set must contain 200 records"
        )

    if len(data["multiclass_train"]) != 800:
        raise ValueError(
            "Multiclass train set must contain 800 records"
        )

    if len(data["multiclass_test"]) != 200:
        raise ValueError(
            "Multiclass test set must contain 200 records"
        )

    print(
        "[OK] Binary split: 800 train / 200 test"
    )

    print(
        "[OK] Multiclass split: 800 train / 200 test"
    )


# ============================================================
# HYBRID VALIDATION
# ============================================================

def validate_hybrid(data):

    print()
    print("========== HYBRID VALIDATION ==========")

    df = data["hybrid"]

    counts = (
        df[
            "hybrid_decision"
        ]
        .value_counts()
        .to_dict()
    )

    expected = {
        "RULE_MATCHED": 800,
        "RULE_CONFIRMED_BY_ML": 120,
        "RULE_EXCEPTION_ML_SUPPORT": 71,
        "RULE_EXCEPTION_ML_DISAGREEMENT": 9,
    }

    for decision, expected_count in expected.items():

        actual = counts.get(
            decision,
            0,
        )

        if actual != expected_count:

            raise ValueError(
                f"{decision}: expected "
                f"{expected_count}, found {actual}"
            )

        print(
            f"[OK] {decision}: {actual}"
        )

    print(
        "[OK] Hybrid decision distribution preserved"
    )

    # Critical rule:
    # deterministic rules remain authoritative.

    matched = df[
        df["hybrid_decision"]
        == "RULE_MATCHED"
    ]

    if len(matched) != 800:

        raise ValueError(
            "Hybrid layer changed deterministic match count"
        )

    print(
        "[OK] Deterministic rules remain authoritative"
    )


# ============================================================
# AGENT WORKFLOW
# ============================================================

def validate_agent(data):

    print()
    print("========== AGENT WORKFLOW VALIDATION ==========")

    decisions = data["decisions"]

    decision_counts = (
        decisions[
            "agent_decision"
        ]
        .value_counts()
        .to_dict()
    )

    expected = {
        "ESCALATE_FOR_REVIEW": 170,
        "DATE_REVIEW": 10,
        "FEE_REVIEW": 10,
        "SETTLEMENT_REVIEW": 10,
    }

    for decision, expected_count in expected.items():

        actual = decision_counts.get(
            decision,
            0,
        )

        if actual != expected_count:

            raise ValueError(
                f"{decision}: expected "
                f"{expected_count}, found {actual}"
            )

        print(
            f"[OK] {decision}: {actual}"
        )

    print(
        "[OK] Agent decision distribution preserved"
    )

    if len(data["actions"]) != 200:

        raise ValueError(
            "Expected 200 agent actions"
        )

    print(
        "[OK] Agent actions: 200"
    )


# ============================================================
# ESCALATION
# ============================================================

def validate_escalations(data):

    print()
    print("========== ESCALATION VALIDATION ==========")

    df = data["escalations"]

    if len(df) != 170:

        raise ValueError(
            f"Expected 170 escalation cases, "
            f"found {len(df)}"
        )

    print(
        "[OK] Escalation cases: 170"
    )

    if "escalation_status" in df.columns:

        statuses = (
            df[
                "escalation_status"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if statuses != ["OPEN"]:

            raise ValueError(
                "Escalation status is not OPEN"
            )

    print(
        "[OK] Escalation cases remain OPEN"
    )


# ============================================================
# AUDIT / REVIEW
# ============================================================

def validate_audit(data):

    print()
    print("========== AUDIT / REVIEW VALIDATION ==========")

    if len(data["audit"]) != 200:
        raise ValueError(
            "Audit count must be 200"
        )

    if len(data["events"]) != 200:
        raise ValueError(
            "Audit event count must be 200"
        )

    if len(data["review"]) != 200:
        raise ValueError(
            "Review count must be 200"
        )

    print(
        "[OK] Audit records: 200"
    )

    print(
        "[OK] Audit events: 200"
    )

    print(
        "[OK] Review cases: 200"
    )

    review = data["review"]

    if "review_status" in review.columns:

        statuses = (
            review[
                "review_status"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if statuses != ["PENDING"]:

            raise ValueError(
                "Review status is not PENDING"
            )

    print(
        "[OK] All review cases remain PENDING"
    )

    events = data["events"]

    if "event_status" in events.columns:

        statuses = (
            events[
                "event_status"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if statuses != ["OPEN"]:

            raise ValueError(
                "Audit events are not OPEN"
            )

    print(
        "[OK] Audit events remain OPEN"
    )


# ============================================================
# DASHBOARD
# ============================================================

def validate_dashboard(data):

    print()
    print("========== DASHBOARD VALIDATION ==========")

    df = data["dashboard"]

    if len(df) != 1000:

        raise ValueError(
            "Dashboard must contain 1000 records"
        )

    matched = (
        df["status"]
        == "MATCHED"
    ).sum()

    exceptions = (
        df["status"]
        == "EXCEPTION"
    ).sum()

    escalated = (
        df["is_escalated"]
        == 1
    ).sum()

    disagreements = (
        df["ml_disagreement"]
        == 1
    ).sum()

    if matched != 800:
        raise ValueError(
            "Dashboard matched count mismatch"
        )

    if exceptions != 200:
        raise ValueError(
            "Dashboard exception count mismatch"
        )

    if escalated != 170:
        raise ValueError(
            "Dashboard escalation count mismatch"
        )

    if disagreements != 9:
        raise ValueError(
            "Dashboard ML disagreement count mismatch"
        )

    print(
        "[OK] Dashboard: 1000 records"
    )

    print(
        "[OK] Dashboard: 800 matched / 200 exceptions"
    )

    print(
        "[OK] Dashboard: 170 escalations"
    )

    print(
        "[OK] Dashboard: 9 ML disagreements"
    )


# ============================================================
# COMPLETE TRANSACTION FLOW
# ============================================================

def validate_flow(data):

    print()
    print("========== COMPLETE TRANSACTION FLOW ==========")

    reconciliation_ids = set(
        data["reconciliation"][
            "transaction_id"
        ]
    )

    dashboard_ids = set(
        data["dashboard"][
            "transaction_id"
        ]
    )

    hybrid_ids = set(
        data["hybrid"][
            "transaction_id"
        ]
    )

    ml_ids = set(
        data["ml_dataset"][
            "transaction_id"
        ]
    )

    if reconciliation_ids != dashboard_ids:

        raise ValueError(
            "Reconciliation → dashboard flow mismatch"
        )

    print(
        "[OK] Reconciliation → Dashboard"
    )

    if reconciliation_ids != hybrid_ids:

        raise ValueError(
            "Reconciliation → hybrid flow mismatch"
        )

    print(
        "[OK] Reconciliation → Hybrid"
    )

    if reconciliation_ids != ml_ids:

        raise ValueError(
            "Reconciliation → ML dataset flow mismatch"
        )

    print(
        "[OK] Reconciliation → ML"
    )

    exception_ids = set(
        data["reconciliation"][
            data["reconciliation"]["status"]
            == "EXCEPTION"
        ]["transaction_id"]
    )

    explanation_ids = set(
        data["explanations"][
            "transaction_id"
        ]
    )

    decision_ids = set(
        data["decisions"][
            "transaction_id"
        ]
    )

    audit_ids = set(
        data["audit"][
            "transaction_id"
        ]
    )

    review_ids = set(
        data["review"][
            "transaction_id"
        ]
    )

    if exception_ids != explanation_ids:

        raise ValueError(
            "Exception → AI explanation flow mismatch"
        )

    print(
        "[OK] Exceptions → AI explanations"
    )

    if exception_ids != decision_ids:

        raise ValueError(
            "Exception → agent decision flow mismatch"
        )

    print(
        "[OK] Exceptions → Agent decisions"
    )

    if exception_ids != audit_ids:

        raise ValueError(
            "Exception → audit flow mismatch"
        )

    print(
        "[OK] Exceptions → Audit"
    )

    if exception_ids != review_ids:

        raise ValueError(
            "Exception → review flow mismatch"
        )

    print(
        "[OK] Exceptions → Review"
    )

    print(
        "[OK] Complete transaction flow preserved"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(data):

    dashboard = data["dashboard"]

    total = len(dashboard)

    matched = (
        dashboard["status"]
        == "MATCHED"
    ).sum()

    exceptions = (
        dashboard["status"]
        == "EXCEPTION"
    ).sum()

    escalations = (
        dashboard["is_escalated"]
        == 1
    ).sum()

    pending = (
        dashboard["is_pending_review"]
        == 1
    ).sum()

    disagreements = (
        dashboard["ml_disagreement"]
        == 1
    ).sum()

    print()
    print("=" * 70)
    print("        FINAL END-TO-END PROJECT SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Total transactions:        {total}"
    )

    print(
        f"Matched transactions:      {matched}"
    )

    print(
        f"Exception transactions:    {exceptions}"
    )

    print(
        f"Reconciliation rate:       {matched / total * 100:.2f}%"
    )

    print()
    print(
        f"AI explanations:            {len(data['explanations'])}"
    )

    print(
        f"Resolution actions:         {len(data['resolution'])}"
    )

    print(
        f"ML-supported exceptions:   {EXPECTED['ml_supported']}"
    )

    print(
        f"ML-confirmed matches:      {EXPECTED['ml_confirmed']}"
    )

    print(
        f"ML disagreements:           {disagreements}"
    )

    print()
    print(
        f"Agent decisions:            {len(data['decisions'])}"
    )

    print(
        f"Agent actions:              {len(data['actions'])}"
    )

    print(
        f"Escalation cases:           {escalations}"
    )

    print(
        f"Pending reviews:            {pending}"
    )

    print()
    print(
        f"Audit records:              {len(data['audit'])}"
    )

    print(
        f"Audit events:               {len(data['events'])}"
    )

    print(
        f"Review cases:               {len(data['review'])}"
    )

    print()
    print("EXECUTION SAFETY")
    print(
        "Deterministic rules:        AUTHORITATIVE"
    )

    print(
        "Agent actions:              SIMULATED"
    )

    print(
        "Escalations:                SIMULATED"
    )

    print(
        "Human review:               REQUIRED"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        data = load_files()

        validate_counts(data)

        validate_transaction_ids(data)

        validate_reconciliation(data)

        validate_evaluation(data)

        validate_exceptions(data)

        validate_severity(data)

        validate_ai(data)

        validate_resolution(data)

        validate_ml_dataset(data)

        validate_ml_splits(data)

        validate_hybrid(data)

        validate_agent(data)

        validate_escalations(data)

        validate_audit(data)

        validate_dashboard(data)

        validate_flow(data)

        final_summary(data)

        print()
        print("=" * 70)
        print("       PHASE 8.1 COMPLETED")
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print("       PHASE 8.1 FAILED")
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()