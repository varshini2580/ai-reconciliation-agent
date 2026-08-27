from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"

DASHBOARD_FILE = RECON_DIR / "dashboard_data.csv"
KPI_FILE = RECON_DIR / "dashboard_kpis.csv"
EXCEPTION_FILE = RECON_DIR / "exception_analytics.csv"
ML_AGENT_FILE = RECON_DIR / "ml_agent_analytics.csv"


# ============================================================
# LOAD
# ============================================================

def load_files():

    print("=" * 70)
    print("        PHASE 7.6 — DASHBOARD VALIDATION")
    print("=" * 70)

    dashboard = pd.read_csv(DASHBOARD_FILE)
    kpis = pd.read_csv(KPI_FILE)
    exceptions = pd.read_csv(EXCEPTION_FILE)
    ml_agent = pd.read_csv(ML_AGENT_FILE)

    print()
    print(f"Dashboard records: {len(dashboard)}")
    print(f"KPI records: {len(kpis)}")
    print(f"Exception analytics records: {len(exceptions)}")
    print(f"ML/Agent analytics records: {len(ml_agent)}")

    print()
    print("[OK] All dashboard files loaded")

    return (
        dashboard,
        kpis,
        exceptions,
        ml_agent,
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_basic(dashboard, kpis, exceptions, ml_agent):

    print()
    print("========== BASIC VALIDATION ==========")

    if len(dashboard) != 1000:
        raise ValueError(
            f"Dashboard should contain 1000 records, "
            f"found {len(dashboard)}"
        )

    print("[OK] Dashboard record count: 1000")

    if dashboard["transaction_id"].duplicated().any():
        raise ValueError(
            "Dashboard transaction IDs are not unique"
        )

    print("[OK] Dashboard transaction IDs unique")

    if dashboard["status"].isin(
        ["MATCHED", "EXCEPTION"]
    ).all() is False:
        raise ValueError(
            "Invalid dashboard status detected"
        )

    print("[OK] Dashboard statuses valid")

    if len(kpis) != 45:
        raise ValueError(
            f"Expected 45 KPI/distribution records, "
            f"found {len(kpis)}"
        )

    print("[OK] KPI dataset contains 45 records")

    if len(exceptions) != 38:
        raise ValueError(
            f"Expected 38 exception analytics records, "
            f"found {len(exceptions)}"
        )

    print("[OK] Exception analytics contains 38 records")

    if len(ml_agent) != 23:
        raise ValueError(
            f"Expected 23 ML/Agent analytics records, "
            f"found {len(ml_agent)}"
        )

    print("[OK] ML/Agent analytics contains 23 records")


# ============================================================
# CORE DASHBOARD COUNTS
# ============================================================

def validate_core_counts(dashboard):

    print()
    print("========== CORE DASHBOARD COUNTS ==========")

    total = len(dashboard)

    matched = (
        dashboard["status"] == "MATCHED"
    ).sum()

    exceptions = (
        dashboard["status"] == "EXCEPTION"
    ).sum()

    escalated = (
        dashboard["is_escalated"] == 1
    ).sum()

    pending_review = (
        dashboard["is_pending_review"] == 1
    ).sum()

    ml_disagreements = (
        dashboard["ml_disagreement"] == 1
    ).sum()

    expected = {
        "Total transactions": (total, 1000),
        "Matched transactions": (matched, 800),
        "Exception transactions": (exceptions, 200),
        "Escalated cases": (escalated, 170),
        "Pending review cases": (pending_review, 200),
        "ML disagreements": (ml_disagreements, 9),
    }

    for name, (actual, expected_value) in expected.items():

        if actual != expected_value:
            raise ValueError(
                f"{name}: expected "
                f"{expected_value}, found {actual}"
            )

        print(
            f"[OK] {name}: {actual}"
        )


# ============================================================
# SEVERITY VALIDATION
# ============================================================

def validate_severity(dashboard):

    print()
    print("========== SEVERITY VALIDATION ==========")

    exceptions = dashboard[
        dashboard["is_exception"] == 1
    ]

    severity_counts = (
        exceptions["severity"]
        .value_counts()
        .to_dict()
    )

    expected = {
        "HIGH": 110,
        "MEDIUM": 70,
        "LOW": 20,
    }

    for severity, expected_count in expected.items():

        actual = severity_counts.get(
            severity,
            0,
        )

        if actual != expected_count:
            raise ValueError(
                f"{severity}: expected "
                f"{expected_count}, found {actual}"
            )

        print(
            f"[OK] {severity}: {actual}"
        )


# ============================================================
# EXCEPTION TYPE VALIDATION
# ============================================================

def validate_exception_types(dashboard):

    print()
    print("========== EXCEPTION TYPE VALIDATION ==========")

    exceptions = dashboard[
        dashboard["is_exception"] == 1
    ]

    actual_counts = (
        exceptions["exception_type"]
        .value_counts()
        .to_dict()
    )

    expected_counts = {
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

    if set(actual_counts.keys()) != set(
        expected_counts.keys()
    ):
        raise ValueError(
            "Dashboard exception types do not match "
            "ground truth"
        )

    for exception_type, expected in expected_counts.items():

        actual = actual_counts[
            exception_type
        ]

        if actual != expected:
            raise ValueError(
                f"{exception_type}: expected "
                f"{expected}, found {actual}"
            )

    print(
        "[OK] All 15 exception types preserved"
    )

    print(
        "[OK] Exception counts match ground truth"
    )


# ============================================================
# HYBRID VALIDATION
# ============================================================

def validate_hybrid(dashboard):

    print()
    print("========== HYBRID VALIDATION ==========")

    actual = (
        dashboard["hybrid_decision"]
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

        count = actual.get(
            decision,
            0,
        )

        if count != expected_count:
            raise ValueError(
                f"{decision}: expected "
                f"{expected_count}, found {count}"
            )

        print(
            f"[OK] {decision}: {count}"
        )

    # --------------------------------------------------------
    # Critical architecture rule:
    # ML must not override deterministic rules.
    # --------------------------------------------------------

    matched_ml_override = dashboard[
        (
            dashboard["status"] == "MATCHED"
        )
        &
        (
            dashboard["hybrid_decision"]
            != "RULE_MATCHED"
        )
    ]

    if len(matched_ml_override) != 0:
        raise ValueError(
            "ML appears to override deterministic "
            "MATCHED results"
        )

    print(
        "[OK] No ML override of deterministic MATCHED results"
    )


# ============================================================
# AGENT VALIDATION
# ============================================================

def validate_agent(dashboard):

    print()
    print("========== AGENT VALIDATION ==========")

    exceptions = dashboard[
        dashboard["is_exception"] == 1
    ]

    agent_counts = (
        exceptions["agent_decision"]
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

        actual = agent_counts.get(
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

    if exceptions[
        "agent_decision"
    ].isna().any():

        raise ValueError(
            "Exception transaction has missing agent decision"
        )

    print(
        "[OK] All exception cases have agent decisions"
    )


# ============================================================
# ESCALATION VALIDATION
# ============================================================

def validate_escalations(dashboard):

    print()
    print("========== ESCALATION VALIDATION ==========")

    escalated = dashboard[
        dashboard["is_escalated"] == 1
    ]

    if len(escalated) != 170:
        raise ValueError(
            f"Expected 170 escalations, "
            f"found {len(escalated)}"
        )

    print(
        "[OK] Escalation count: 170"
    )

    if not (
        escalated["requires_human"]
        .astype(str)
        .str.upper()
        .eq("YES")
        .all()
    ):
        raise ValueError(
            "Escalated cases must require human review"
        )

    print(
        "[OK] All escalations require human review"
    )

    execution_modes = (
        dashboard["execution_mode"]
        .dropna()
        .unique()
        .tolist()
    )

    if execution_modes != ["SIMULATED"]:

        raise ValueError(
            "Dashboard contains non-SIMULATED execution mode"
        )

    print(
        "[OK] Execution mode = SIMULATED"
    )


# ============================================================
# REVIEW VALIDATION
# ============================================================

def validate_reviews(dashboard):

    print()
    print("========== REVIEW VALIDATION ==========")

    pending = dashboard[
        dashboard["is_pending_review"] == 1
    ]

    if len(pending) != 200:
        raise ValueError(
            f"Expected 200 pending reviews, "
            f"found {len(pending)}"
        )

    print(
        "[OK] Pending review count: 200"
    )

    if not (
        pending["review_status"]
        .astype(str)
        .str.upper()
        .eq("PENDING")
        .all()
    ):
        raise ValueError(
            "Not all review cases are PENDING"
        )

    print(
        "[OK] All review cases remain PENDING"
    )

    if not (
        pending["requires_human_review"]
        .astype(str)
        .str.upper()
        .eq("YES")
        .all()
    ):
        raise ValueError(
            "Pending review cases must require human review"
        )

    print(
        "[OK] Human-review requirement preserved"
    )


# ============================================================
# KPI VALIDATION
# ============================================================

def validate_kpis(kpis):

    print()
    print("========== KPI VALIDATION ==========")

    kpi_rows = kpis[
        kpis["metric_type"] == "KPI"
    ].copy()

    if len(kpi_rows) != 17:
        raise ValueError(
            f"Expected 17 KPI rows, "
            f"found {len(kpi_rows)}"
        )

    values = dict(
        zip(
            kpi_rows["kpi_name"],
            kpi_rows["kpi_value"],
        )
    )

    expected = {
        "TOTAL_TRANSACTIONS": 1000,
        "MATCHED_TRANSACTIONS": 800,
        "EXCEPTION_TRANSACTIONS": 200,
        "RECONCILIATION_RATE_PERCENT": 80.0,
        "EXCEPTION_RATE_PERCENT": 20.0,
        "ESCALATED_CASES": 170,
        "ESCALATION_RATE_PERCENT": 17.0,
        "PENDING_REVIEW_CASES": 200,
        "PENDING_REVIEW_RATE_PERCENT": 20.0,
        "HIGH_SEVERITY_EXCEPTIONS": 110,
        "MEDIUM_SEVERITY_EXCEPTIONS": 70,
        "LOW_SEVERITY_EXCEPTIONS": 20,
        "ML_SUPPORTED_EXCEPTIONS": 71,
        "ML_SUPPORT_RATE_PERCENT": 35.5,
        "ML_DISAGREEMENTS": 9,
        "ML_DISAGREEMENT_RATE_PERCENT": 4.5,
        "ML_CONFIRMED_RULE_MATCHES": 120,
    }

    for name, expected_value in expected.items():

        actual = values.get(name)

        if actual != expected_value:
            raise ValueError(
                f"{name}: expected "
                f"{expected_value}, found {actual}"
            )

    print(
        "[OK] All 17 dashboard KPIs match validated results"
    )


# ============================================================
# FINANCIAL VALIDATION
# ============================================================

def validate_financial_analytics(
    dashboard,
    exception_analytics,
):

    print()
    print("========== FINANCIAL ANALYTICS VALIDATION ==========")

    dashboard_exceptions = dashboard[
        dashboard["is_exception"] == 1
    ].copy()

    dashboard_exceptions[
        "difference"
    ] = pd.to_numeric(
        dashboard_exceptions[
            "difference"
        ],
        errors="coerce",
    )

    dashboard_total = round(
        dashboard_exceptions[
            "difference"
        ].sum(),
        2,
    )

    analytics_rows = exception_analytics[
        exception_analytics[
            "analytics_category"
        ]
        == "EXCEPTION_TYPE"
    ]

    analytics_total = round(
        analytics_rows[
            "total_difference"
        ].sum(),
        2,
    )

    if dashboard_total != analytics_total:
        raise ValueError(
            "Financial difference mismatch between "
            "dashboard and exception analytics"
        )

    print(
        f"[OK] Financial total preserved: "
        f"{dashboard_total:,.2f}"
    )


# ============================================================
# FILTER-SAFETY VALIDATION
# ============================================================

def validate_filter_safety(dashboard):

    print()
    print("========== FILTER-SAFETY VALIDATION ==========")

    # --------------------------------------------------------
    # Simulate the same basic filters used by Streamlit.
    # --------------------------------------------------------

    matched = dashboard[
        dashboard["status"] == "MATCHED"
    ]

    exceptions = dashboard[
        dashboard["status"] == "EXCEPTION"
    ]

    if len(matched) != 800:
        raise ValueError(
            "Status filter does not return 800 MATCHED records"
        )

    if len(exceptions) != 200:
        raise ValueError(
            "Status filter does not return 200 EXCEPTION records"
        )

    print(
        "[OK] Status filtering preserves 800 / 200 split"
    )

    high = dashboard[
        dashboard["severity"] == "HIGH"
    ]

    medium = dashboard[
        dashboard["severity"] == "MEDIUM"
    ]

    low = dashboard[
        dashboard["severity"] == "LOW"
    ]

    if len(high) != 110:
        raise ValueError(
            "HIGH severity filter mismatch"
        )

    if len(medium) != 70:
        raise ValueError(
            "MEDIUM severity filter mismatch"
        )

    if len(low) != 20:
        raise ValueError(
            "LOW severity filter mismatch"
        )

    print(
        "[OK] Severity filtering preserves 110 / 70 / 20"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(dashboard):

    total = len(dashboard)

    matched = (
        dashboard["status"] == "MATCHED"
    ).sum()

    exceptions = (
        dashboard["status"] == "EXCEPTION"
    ).sum()

    escalations = (
        dashboard["is_escalated"] == 1
    ).sum()

    reviews = (
        dashboard["is_pending_review"] == 1
    ).sum()

    ml_disagreements = (
        dashboard["ml_disagreement"] == 1
    ).sum()

    print()
    print("=" * 70)
    print("        PHASE 7 DASHBOARD VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Total transactions:       {total}"
    )

    print(
        f"Matched transactions:     {matched}"
    )

    print(
        f"Exception transactions:   {exceptions}"
    )

    print(
        f"Escalation cases:         {escalations}"
    )

    print(
        f"Pending review cases:     {reviews}"
    )

    print(
        f"ML disagreements:         {ml_disagreements}"
    )

    print()
    print(
        "Dashboard status: VALIDATED"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        (
            dashboard,
            kpis,
            exceptions,
            ml_agent,
        ) = load_files()

        validate_basic(
            dashboard,
            kpis,
            exceptions,
            ml_agent,
        )

        validate_core_counts(
            dashboard
        )

        validate_severity(
            dashboard
        )

        validate_exception_types(
            dashboard
        )

        validate_hybrid(
            dashboard
        )

        validate_agent(
            dashboard
        )

        validate_escalations(
            dashboard
        )

        validate_reviews(
            dashboard
        )

        validate_kpis(
            kpis
        )

        validate_financial_analytics(
            dashboard,
            exceptions,
        )

        validate_filter_safety(
            dashboard
        )

        print_summary(
            dashboard
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 7.6 COMPLETED"
        )
        print(
            "       PHASE 7 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 7.6 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()