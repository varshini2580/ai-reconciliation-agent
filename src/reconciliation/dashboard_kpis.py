from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"

INPUT_FILE = RECON_DIR / "dashboard_data.csv"

OUTPUT_FILE = RECON_DIR / "dashboard_kpis.csv"


# ============================================================
# LOAD
# ============================================================

def load_dashboard_data():

    print("=" * 70)
    print("        PHASE 7.2 — KPI / SUMMARY GENERATION")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dashboard data not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Dashboard records: {len(df)}"
    )

    return df


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(df):

    print()
    print(
        "========== INPUT VALIDATION =========="
    )

    required_columns = [
        "transaction_id",
        "status",
        "is_exception",
        "is_escalated",
        "is_pending_review",
        "ml_supported_exception",
        "ml_disagreement",
        "ml_confirmed_rule_match",
        "exception_type",
        "severity",
        "hybrid_decision",
        "agent_decision",
        "escalation_required",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing dashboard columns: "
            + ", ".join(missing)
        )

    print(
        "[OK] Dashboard schema valid"
    )

    if len(df) != 1000:

        raise ValueError(
            f"Expected 1000 records, "
            f"found {len(df)}"
        )

    print(
        "[OK] 1000 dashboard records loaded"
    )

    if df[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected"
        )

    print(
        "[OK] Transaction IDs unique"
    )


# ============================================================
# KPI HELPER
# ============================================================

def add_kpi(kpis, name, value):

    kpis.append(
        {
            "kpi_name": name,
            "kpi_value": value,
        }
    )


# ============================================================
# BUILD KPIs
# ============================================================

def build_kpis(df):

    print()
    print(
        "========== BUILDING DASHBOARD KPIs =========="
    )

    total_transactions = len(df)

    matched = (
        df["status"] == "MATCHED"
    ).sum()

    exceptions = (
        df["status"] == "EXCEPTION"
    ).sum()

    escalations = (
        df["is_escalated"] == 1
    ).sum()

    pending_reviews = (
        df["is_pending_review"] == 1
    ).sum()

    high_exceptions = (
        df["severity"] == "HIGH"
    ).sum()

    medium_exceptions = (
        df["severity"] == "MEDIUM"
    ).sum()

    low_exceptions = (
        df["severity"] == "LOW"
    ).sum()

    ml_supported = (
        df["ml_supported_exception"] == 1
    ).sum()

    ml_disagreements = (
        df["ml_disagreement"] == 1
    ).sum()

    ml_confirmed_matches = (
        df["ml_confirmed_rule_match"] == 1
    ).sum()

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    reconciliation_rate = (
        matched / total_transactions * 100
    )

    exception_rate = (
        exceptions / total_transactions * 100
    )

    escalation_rate = (
        escalations / total_transactions * 100
    )

    pending_review_rate = (
        pending_reviews / total_transactions * 100
    )

    ml_support_rate = (
        ml_supported / exceptions * 100
    )

    ml_disagreement_rate = (
        ml_disagreements / exceptions * 100
    )

    # --------------------------------------------------------
    # KPI records
    # --------------------------------------------------------

    kpis = []

    add_kpi(
        kpis,
        "TOTAL_TRANSACTIONS",
        total_transactions,
    )

    add_kpi(
        kpis,
        "MATCHED_TRANSACTIONS",
        matched,
    )

    add_kpi(
        kpis,
        "EXCEPTION_TRANSACTIONS",
        exceptions,
    )

    add_kpi(
        kpis,
        "RECONCILIATION_RATE_PERCENT",
        round(reconciliation_rate, 2),
    )

    add_kpi(
        kpis,
        "EXCEPTION_RATE_PERCENT",
        round(exception_rate, 2),
    )

    add_kpi(
        kpis,
        "ESCALATED_CASES",
        escalations,
    )

    add_kpi(
        kpis,
        "ESCALATION_RATE_PERCENT",
        round(escalation_rate, 2),
    )

    add_kpi(
        kpis,
        "PENDING_REVIEW_CASES",
        pending_reviews,
    )

    add_kpi(
        kpis,
        "PENDING_REVIEW_RATE_PERCENT",
        round(pending_review_rate, 2),
    )

    add_kpi(
        kpis,
        "HIGH_SEVERITY_EXCEPTIONS",
        high_exceptions,
    )

    add_kpi(
        kpis,
        "MEDIUM_SEVERITY_EXCEPTIONS",
        medium_exceptions,
    )

    add_kpi(
        kpis,
        "LOW_SEVERITY_EXCEPTIONS",
        low_exceptions,
    )

    add_kpi(
        kpis,
        "ML_SUPPORTED_EXCEPTIONS",
        ml_supported,
    )

    add_kpi(
        kpis,
        "ML_SUPPORT_RATE_PERCENT",
        round(ml_support_rate, 2),
    )

    add_kpi(
        kpis,
        "ML_DISAGREEMENTS",
        ml_disagreements,
    )

    add_kpi(
        kpis,
        "ML_DISAGREEMENT_RATE_PERCENT",
        round(ml_disagreement_rate, 2),
    )

    add_kpi(
        kpis,
        "ML_CONFIRMED_RULE_MATCHES",
        ml_confirmed_matches,
    )

    result = pd.DataFrame(kpis)

    print(
        f"KPI records created: {len(result)}"
    )

    return result


# ============================================================
# BUILD DISTRIBUTION TABLES
# ============================================================

def build_distributions(df):

    print()
    print(
        "========== BUILDING DASHBOARD DISTRIBUTIONS =========="
    )

    # --------------------------------------------------------
    # Exception distribution
    # --------------------------------------------------------

    exception_distribution = (
        df[
            df["is_exception"] == 1
        ]
        .groupby(
            "exception_type"
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    exception_distribution[
        "metric"
    ] = "EXCEPTION_TYPE"

    # --------------------------------------------------------
    # Severity distribution
    # --------------------------------------------------------

    severity_distribution = (
        df[
            df["is_exception"] == 1
        ]
        .groupby(
            "severity"
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    severity_distribution[
        "metric"
    ] = "SEVERITY"

    severity_distribution = (
        severity_distribution.rename(
            columns={
                "severity": "category"
            }
        )
    )

    exception_distribution = (
        exception_distribution.rename(
            columns={
                "exception_type": "category"
            }
        )
    )

    # --------------------------------------------------------
    # Hybrid distribution
    # --------------------------------------------------------

    hybrid_distribution = (
        df.groupby(
            "hybrid_decision"
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    hybrid_distribution[
        "metric"
    ] = "HYBRID_DECISION"

    hybrid_distribution = (
        hybrid_distribution.rename(
            columns={
                "hybrid_decision": "category"
            }
        )
    )

    # --------------------------------------------------------
    # Agent decision distribution
    #
    # Exclude matched transactions because they have no
    # agent decision.
    # --------------------------------------------------------

    agent_distribution = (
        df[
            df["is_exception"] == 1
        ]
        .groupby(
            "agent_decision"
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    agent_distribution[
        "metric"
    ] = "AGENT_DECISION"

    agent_distribution = (
        agent_distribution.rename(
            columns={
                "agent_decision": "category"
            }
        )
    )

    # --------------------------------------------------------
    # Escalation distribution
    # --------------------------------------------------------

    escalation_distribution = (
        df[
            df["is_exception"] == 1
        ]
        .groupby(
            "escalation_required"
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    escalation_distribution[
        "metric"
    ] = "ESCALATION"

    escalation_distribution = (
        escalation_distribution.rename(
            columns={
                "escalation_required": "category"
            }
        )
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    distribution = pd.concat(
        [
            exception_distribution[
                [
                    "metric",
                    "category",
                    "count",
                ]
            ],
            severity_distribution[
                [
                    "metric",
                    "category",
                    "count",
                ]
            ],
            hybrid_distribution[
                [
                    "metric",
                    "category",
                    "count",
                ]
            ],
            agent_distribution[
                [
                    "metric",
                    "category",
                    "count",
                ]
            ],
            escalation_distribution[
                [
                    "metric",
                    "category",
                    "count",
                ]
            ],
        ],
        ignore_index=True,
    )

    print(
        f"Distribution records created: "
        f"{len(distribution)}"
    )

    return distribution


# ============================================================
# VALIDATE KPIs
# ============================================================

def validate_kpis(kpis, distributions):

    print()
    print("=" * 70)
    print("        PHASE 7.2 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # KPI count
    # --------------------------------------------------------

    if len(kpis) != 17:

        raise ValueError(
            f"Expected 17 KPI records, "
            f"found {len(kpis)}"
        )

    print(
        "[OK] KPI record count: 17"
    )

    # --------------------------------------------------------
    # KPI uniqueness
    # --------------------------------------------------------

    if kpis[
        "kpi_name"
    ].duplicated().any():

        raise ValueError(
            "Duplicate KPI names detected"
        )

    print(
        "[OK] KPI names unique"
    )

    # --------------------------------------------------------
    # Extract KPI values
    # --------------------------------------------------------

    values = dict(
        zip(
            kpis["kpi_name"],
            kpis["kpi_value"],
        )
    )

    # --------------------------------------------------------
    # Core counts
    # --------------------------------------------------------

    expected = {
        "TOTAL_TRANSACTIONS": 1000,
        "MATCHED_TRANSACTIONS": 800,
        "EXCEPTION_TRANSACTIONS": 200,
        "ESCALATED_CASES": 170,
        "PENDING_REVIEW_CASES": 200,
        "HIGH_SEVERITY_EXCEPTIONS": 110,
        "MEDIUM_SEVERITY_EXCEPTIONS": 70,
        "LOW_SEVERITY_EXCEPTIONS": 20,
        "ML_SUPPORTED_EXCEPTIONS": 71,
        "ML_DISAGREEMENTS": 9,
        "ML_CONFIRMED_RULE_MATCHES": 120,
    }

    for name, expected_value in expected.items():

        actual = values[name]

        if actual != expected_value:

            raise ValueError(
                f"{name}: expected "
                f"{expected_value}, found {actual}"
            )

        print(
            f"[OK] {name}: {actual}"
        )

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    rate_checks = {
        "RECONCILIATION_RATE_PERCENT": 80.0,
        "EXCEPTION_RATE_PERCENT": 20.0,
        "ESCALATION_RATE_PERCENT": 17.0,
        "PENDING_REVIEW_RATE_PERCENT": 20.0,
        "ML_SUPPORT_RATE_PERCENT": 35.5,
        "ML_DISAGREEMENT_RATE_PERCENT": 4.5,
    }

    for name, expected_value in rate_checks.items():

        actual = values[name]

        if actual != expected_value:

            raise ValueError(
                f"{name}: expected "
                f"{expected_value}, found {actual}"
            )

        print(
            f"[OK] {name}: {actual}%"
        )

    # --------------------------------------------------------
    # Distribution validation
    # --------------------------------------------------------

    exception_counts = distributions[
        distributions["metric"]
        == "EXCEPTION_TYPE"
    ]["count"].sum()

    if exception_counts != 200:

        raise ValueError(
            "Exception distribution does not total 200"
        )

    print(
        "[OK] Exception distribution totals 200"
    )

    severity_counts = distributions[
        distributions["metric"]
        == "SEVERITY"
    ]["count"].sum()

    if severity_counts != 200:

        raise ValueError(
            "Severity distribution does not total 200"
        )

    print(
        "[OK] Severity distribution totals 200"
    )

    hybrid_counts = distributions[
        distributions["metric"]
        == "HYBRID_DECISION"
    ]["count"].sum()

    if hybrid_counts != 1000:

        raise ValueError(
            "Hybrid distribution does not total 1000"
        )

    print(
        "[OK] Hybrid distribution totals 1000"
    )

    agent_counts = distributions[
        distributions["metric"]
        == "AGENT_DECISION"
    ]["count"].sum()

    if agent_counts != 200:

        raise ValueError(
            "Agent distribution does not total 200"
        )

    print(
        "[OK] Agent distribution totals 200"
    )

    escalation_counts = distributions[
        distributions["metric"]
        == "ESCALATION"
    ]["count"].sum()

    if escalation_counts != 200:

        raise ValueError(
            "Escalation distribution does not total 200"
        )

    print(
        "[OK] Escalation distribution totals 200"
    )


# ============================================================
# SAVE KPI + DISTRIBUTIONS
# ============================================================

def save_output(kpis, distributions):

    # --------------------------------------------------------
    # Store both KPI and distribution rows in one file.
    #
    # KPI rows have metric_type = KPI.
    # Distribution rows have metric_type = DISTRIBUTION.
    # --------------------------------------------------------

    kpi_output = kpis.copy()

    kpi_output[
        "metric_type"
    ] = "KPI"

    kpi_output[
        "category"
    ] = pd.NA

    kpi_output[
        "count"
    ] = pd.NA

    kpi_output = kpi_output[
        [
            "metric_type",
            "kpi_name",
            "kpi_value",
            "category",
            "count",
        ]
    ]

    distribution_output = distributions.copy()

    distribution_output[
        "metric_type"
    ] = "DISTRIBUTION"

    distribution_output[
        "kpi_name"
    ] = pd.NA

    distribution_output[
        "kpi_value"
    ] = pd.NA

    distribution_output = distribution_output[
        [
            "metric_type",
            "kpi_name",
            "kpi_value",
            "category",
            "count",
        ]
    ]

    output = pd.concat(
        [
            kpi_output,
            distribution_output,
        ],
        ignore_index=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        df = load_dashboard_data()

        validate_input(
            df
        )

        kpis = build_kpis(
            df
        )

        distributions = build_distributions(
            df
        )

        validate_kpis(
            kpis,
            distributions,
        )

        output = save_output(
            kpis,
            distributions,
        )

        print()
        print(
            "Dashboard KPI data saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print(
            f"Total dashboard metric records: "
            f"{len(output)}"
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