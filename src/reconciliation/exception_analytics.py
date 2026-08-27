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

OUTPUT_FILE = RECON_DIR / "exception_analytics.csv"


# ============================================================
# LOAD
# ============================================================

def load_dashboard_data():

    print("=" * 70)
    print("        PHASE 7.3 — EXCEPTION ANALYTICS")
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
        "exception_type",
        "severity",
        "difference",
        "is_exception",
        "is_escalated",
        "is_pending_review",
        "ml_supported_exception",
        "ml_disagreement",
        "escalation_required",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
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
# BUILD EXCEPTION ANALYTICS
# ============================================================

def build_exception_analytics(df):

    print()
    print(
        "========== BUILDING EXCEPTION ANALYTICS =========="
    )

    # --------------------------------------------------------
    # Only exception transactions belong in this analysis.
    # --------------------------------------------------------

    exceptions = df[
        df["is_exception"] == 1
    ].copy()

    if len(exceptions) != 200:

        raise ValueError(
            f"Expected 200 exception records, "
            f"found {len(exceptions)}"
        )

    print(
        f"Exception records analyzed: {len(exceptions)}"
    )

    # --------------------------------------------------------
    # Ensure numeric financial difference.
    # --------------------------------------------------------

    exceptions[
        "difference"
    ] = pd.to_numeric(
        exceptions["difference"],
        errors="coerce",
    )

    if exceptions[
        "difference"
    ].isna().any():

        raise ValueError(
            "Missing or invalid financial difference"
        )

    # --------------------------------------------------------
    # Group by exception type.
    # --------------------------------------------------------

    result = (
        exceptions
        .groupby("exception_type")
        .agg(
            exception_count=(
                "transaction_id",
                "count",
            ),

            total_difference=(
                "difference",
                "sum",
            ),

            average_difference=(
                "difference",
                "mean",
            ),

            maximum_difference=(
                "difference",
                "max",
            ),

            escalated_cases=(
                "is_escalated",
                "sum",
            ),

            pending_review_cases=(
                "is_pending_review",
                "sum",
            ),

            ml_supported_cases=(
                "ml_supported_exception",
                "sum",
            ),

            ml_disagreement_cases=(
                "ml_disagreement",
                "sum",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Severity count by exception type.
    # --------------------------------------------------------

    severity_counts = (
        pd.crosstab(
            exceptions["exception_type"],
            exceptions["severity"],
        )
        .reset_index()
    )

    # Ensure all expected severity columns exist.
    for severity in [
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:

        if severity not in severity_counts.columns:

            severity_counts[
                severity
            ] = 0

    severity_counts = severity_counts[
        [
            "exception_type",
            "HIGH",
            "MEDIUM",
            "LOW",
        ]
    ]

    severity_counts = severity_counts.rename(
        columns={
            "HIGH": "high_severity_count",
            "MEDIUM": "medium_severity_count",
            "LOW": "low_severity_count",
        }
    )

    result = result.merge(
        severity_counts,
        on="exception_type",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Rates.
    # --------------------------------------------------------

    result[
        "escalation_rate_percent"
    ] = (
        result["escalated_cases"]
        / result["exception_count"]
        * 100
    ).round(2)

    result[
        "ml_support_rate_percent"
    ] = (
        result["ml_supported_cases"]
        / result["exception_count"]
        * 100
    ).round(2)

    result[
        "ml_disagreement_rate_percent"
    ] = (
        result["ml_disagreement_cases"]
        / result["exception_count"]
        * 100
    ).round(2)

    result[
        "review_rate_percent"
    ] = (
        result["pending_review_cases"]
        / result["exception_count"]
        * 100
    ).round(2)

    # --------------------------------------------------------
    # Sort by exception frequency.
    # --------------------------------------------------------

    result = result.sort_values(
        by=[
            "exception_count",
            "total_difference",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Rank.
    # --------------------------------------------------------

    result.insert(
        0,
        "exception_rank",
        range(
            1,
            len(result) + 1,
        ),
    )

    # --------------------------------------------------------
    # Dashboard category.
    # --------------------------------------------------------

    result[
        "analytics_category"
    ] = "EXCEPTION_TYPE"

    print(
        f"Exception types analyzed: {len(result)}"
    )

    return result


# ============================================================
# BUILD SEVERITY ANALYTICS
# ============================================================

def build_severity_analytics(df):

    print()
    print(
        "========== BUILDING SEVERITY ANALYTICS =========="
    )

    exceptions = df[
        df["is_exception"] == 1
    ].copy()

    result = (
        exceptions
        .groupby("severity")
        .agg(
            exception_count=(
                "transaction_id",
                "count",
            ),

            total_difference=(
                "difference",
                "sum",
            ),

            average_difference=(
                "difference",
                "mean",
            ),

            escalated_cases=(
                "is_escalated",
                "sum",
            ),

            pending_review_cases=(
                "is_pending_review",
                "sum",
            ),
        )
        .reset_index()
    )

    result[
        "escalation_rate_percent"
    ] = (
        result["escalated_cases"]
        / result["exception_count"]
        * 100
    ).round(2)

    result[
        "review_rate_percent"
    ] = (
        result["pending_review_cases"]
        / result["exception_count"]
        * 100
    ).round(2)

    result[
        "analytics_category"
    ] = "SEVERITY"

    result = result.sort_values(
        by="exception_count",
        ascending=False,
    ).reset_index(
        drop=True
    )

    print(
        f"Severity categories analyzed: {len(result)}"
    )

    return result


# ============================================================
# BUILD OVERALL FINANCIAL IMPACT
# ============================================================

def build_financial_summary(df):

    print()
    print(
        "========== BUILDING FINANCIAL IMPACT SUMMARY =========="
    )

    exceptions = df[
        df["is_exception"] == 1
    ].copy()

    differences = pd.to_numeric(
        exceptions["difference"],
        errors="coerce",
    )

    total_difference = differences.sum()

    absolute_difference = (
        differences.abs().sum()
    )

    average_difference = (
        differences.mean()
    )

    average_absolute_difference = (
        differences.abs().mean()
    )

    maximum_absolute_difference = (
        differences.abs().max()
    )

    result = pd.DataFrame(
        [
            {
                "analytics_category": "FINANCIAL_SUMMARY",
                "metric": "TOTAL_DIFFERENCE",
                "value": round(
                    total_difference,
                    2,
                ),
            },
            {
                "analytics_category": "FINANCIAL_SUMMARY",
                "metric": "ABSOLUTE_DIFFERENCE",
                "value": round(
                    absolute_difference,
                    2,
                ),
            },
            {
                "analytics_category": "FINANCIAL_SUMMARY",
                "metric": "AVERAGE_DIFFERENCE",
                "value": round(
                    average_difference,
                    2,
                ),
            },
            {
                "analytics_category": "FINANCIAL_SUMMARY",
                "metric": "AVERAGE_ABSOLUTE_DIFFERENCE",
                "value": round(
                    average_absolute_difference,
                    2,
                ),
            },
            {
                "analytics_category": "FINANCIAL_SUMMARY",
                "metric": "MAXIMUM_ABSOLUTE_DIFFERENCE",
                "value": round(
                    maximum_absolute_difference,
                    2,
                ),
            },
        ]
    )

    print(
        "[OK] Financial impact metrics created"
    )

    return result


# ============================================================
# BUILD ESCALATION ANALYTICS
# ============================================================

def build_escalation_analytics(df):

    print()
    print(
        "========== BUILDING ESCALATION ANALYTICS =========="
    )

    exceptions = df[
        df["is_exception"] == 1
    ].copy()

    result = (
        exceptions
        .groupby("exception_type")
        .agg(
            total_cases=(
                "transaction_id",
                "count",
            ),

            escalated_cases=(
                "is_escalated",
                "sum",
            ),
        )
        .reset_index()
    )

    result[
        "escalation_rate_percent"
    ] = (
        result["escalated_cases"]
        / result["total_cases"]
        * 100
    ).round(2)

    result[
        "analytics_category"
    ] = "ESCALATION_BY_EXCEPTION"

    result = result.sort_values(
        by=[
            "escalated_cases",
            "total_cases",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    print(
        f"Escalation categories analyzed: {len(result)}"
    )

    return result


# ============================================================
# VALIDATE
# ============================================================

def validate_analytics(
    exception_analytics,
    severity_analytics,
    financial_summary,
    escalation_analytics,
):

    print()
    print("=" * 70)
    print(
        "        PHASE 7.3 VALIDATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Exception types
    # --------------------------------------------------------

    if len(exception_analytics) != 15:

        raise ValueError(
            f"Expected 15 exception types, "
            f"found {len(exception_analytics)}"
        )

    print(
        "[OK] Exception types: 15"
    )

    # --------------------------------------------------------
    # Total exception count
    # --------------------------------------------------------

    total_exception_count = (
        exception_analytics[
            "exception_count"
        ].sum()
    )

    if total_exception_count != 200:

        raise ValueError(
            f"Exception analytics total should be 200, "
            f"found {total_exception_count}"
        )

    print(
        "[OK] Exception analytics total: 200"
    )

    # --------------------------------------------------------
    # Escalations
    # --------------------------------------------------------

    total_escalations = (
        exception_analytics[
            "escalated_cases"
        ].sum()
    )

    if total_escalations != 170:

        raise ValueError(
            f"Expected 170 escalations, "
            f"found {total_escalations}"
        )

    print(
        "[OK] Escalation analytics total: 170"
    )

    # --------------------------------------------------------
    # Pending reviews
    # --------------------------------------------------------

    total_reviews = (
        exception_analytics[
            "pending_review_cases"
        ].sum()
    )

    if total_reviews != 200:

        raise ValueError(
            f"Expected 200 pending reviews, "
            f"found {total_reviews}"
        )

    print(
        "[OK] Pending review analytics total: 200"
    )

    # --------------------------------------------------------
    # ML support
    # --------------------------------------------------------

    total_ml_support = (
        exception_analytics[
            "ml_supported_cases"
        ].sum()
    )

    if total_ml_support != 71:

        raise ValueError(
            f"Expected 71 ML-supported exceptions, "
            f"found {total_ml_support}"
        )

    print(
        "[OK] ML-supported cases: 71"
    )

    # --------------------------------------------------------
    # ML disagreement
    # --------------------------------------------------------

    total_ml_disagreement = (
        exception_analytics[
            "ml_disagreement_cases"
        ].sum()
    )

    if total_ml_disagreement != 9:

        raise ValueError(
            f"Expected 9 ML disagreements, "
            f"found {total_ml_disagreement}"
        )

    print(
        "[OK] ML disagreements: 9"
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    if len(severity_analytics) != 3:

        raise ValueError(
            "Expected HIGH, MEDIUM and LOW severity"
        )

    severity_total = (
        severity_analytics[
            "exception_count"
        ].sum()
    )

    if severity_total != 200:

        raise ValueError(
            f"Severity total should be 200, "
            f"found {severity_total}"
        )

    print(
        "[OK] Severity analytics total: 200"
    )

    # --------------------------------------------------------
    # Financial summary
    # --------------------------------------------------------

    if len(financial_summary) != 5:

        raise ValueError(
            "Expected 5 financial summary metrics"
        )

    print(
        "[OK] Financial impact metrics: 5"
    )

    # --------------------------------------------------------
    # Escalation analytics
    # --------------------------------------------------------

    if len(escalation_analytics) != 15:

        raise ValueError(
            "Expected escalation analytics "
            "for 15 exception types"
        )

    escalation_total = (
        escalation_analytics[
            "escalated_cases"
        ].sum()
    )

    if escalation_total != 170:

        raise ValueError(
            f"Escalation total should be 170, "
            f"found {escalation_total}"
        )

    print(
        "[OK] Escalation-by-exception total: 170"
    )

    # --------------------------------------------------------
    # Required exception counts
    # --------------------------------------------------------

    expected_counts = {
        "AMOUNT_MISMATCH": 30,
        "MISSING_SETTLEMENT": 20,
        "MISSING_PAYMENT": 20,
        "DUPLICATE_PAYMENT": 15,
        "FAILED_PAYMENT": 15,
        "REFUND": 15,
        "PARTIAL_SETTLEMENT": 15,
        "INCORRECT_FEE": 10,
        "DATE_MISMATCH": 10,
        "DUPLICATE_SETTLEMENT": 10,
        "WRONG_TRANSACTION_REFERENCE": 10,
        "CHARGEBACK": 10,
        "SETTLEMENT_DELAY": 10,
        "MULTIPLE_PAYMENTS": 5,
        "UNKNOWN_ADJUSTMENT": 5,
    }

    actual_counts = dict(
        zip(
            exception_analytics[
                "exception_type"
            ],
            exception_analytics[
                "exception_count"
            ],
        )
    )

    for exception_type, expected in expected_counts.items():

        actual = actual_counts.get(
            exception_type
        )

        if actual != expected:

            raise ValueError(
                f"{exception_type}: expected "
                f"{expected}, found {actual}"
            )

    print(
        "[OK] Exception-type counts match ground truth"
    )


# ============================================================
# SAVE
# ============================================================

def save_output(
    exception_analytics,
    severity_analytics,
    financial_summary,
    escalation_analytics,
):

    # --------------------------------------------------------
    # Convert different analytics tables into one dashboard
    # analytics file.
    # --------------------------------------------------------

    exception_output = exception_analytics.copy()

    exception_output[
        "metric"
    ] = "EXCEPTION_TYPE"

    exception_output[
        "category"
    ] = exception_output[
        "exception_type"
    ]

    exception_output[
        "value"
    ] = exception_output[
        "exception_count"
    ]

    exception_output = exception_output[
        [
            "analytics_category",
            "metric",
            "category",
            "value",
            "exception_count",
            "total_difference",
            "average_difference",
            "maximum_difference",
            "high_severity_count",
            "medium_severity_count",
            "low_severity_count",
            "escalated_cases",
            "pending_review_cases",
            "ml_supported_cases",
            "ml_disagreement_cases",
            "escalation_rate_percent",
            "ml_support_rate_percent",
            "ml_disagreement_rate_percent",
            "review_rate_percent",
        ]
    ]

    severity_output = severity_analytics.copy()

    severity_output[
        "metric"
    ] = "SEVERITY"

    severity_output[
        "category"
    ] = severity_output[
        "severity"
    ]

    severity_output[
        "value"
    ] = severity_output[
        "exception_count"
    ]

    # Add fields that don't apply to severity rows.
    for column in [
        "maximum_difference",
        "high_severity_count",
        "medium_severity_count",
        "low_severity_count",
        "ml_supported_cases",
        "ml_disagreement_cases",
        "ml_support_rate_percent",
        "ml_disagreement_rate_percent",
    ]:

        severity_output[
            column
        ] = pd.NA

    severity_output = severity_output[
        [
            "analytics_category",
            "metric",
            "category",
            "value",
            "exception_count",
            "total_difference",
            "average_difference",
            "maximum_difference",
            "high_severity_count",
            "medium_severity_count",
            "low_severity_count",
            "escalated_cases",
            "pending_review_cases",
            "ml_supported_cases",
            "ml_disagreement_cases",
            "escalation_rate_percent",
            "ml_support_rate_percent",
            "ml_disagreement_rate_percent",
            "review_rate_percent",
        ]
    ]

    financial_output = financial_summary.copy()

    financial_output[
        "category"
    ] = financial_output[
        "metric"
    ]

    financial_output[
        "value"
    ] = financial_output[
        "value"
    ]

    for column in [
        "exception_count",
        "total_difference",
        "average_difference",
        "maximum_difference",
        "high_severity_count",
        "medium_severity_count",
        "low_severity_count",
        "escalated_cases",
        "pending_review_cases",
        "ml_supported_cases",
        "ml_disagreement_cases",
        "escalation_rate_percent",
        "ml_support_rate_percent",
        "ml_disagreement_rate_percent",
        "review_rate_percent",
    ]:

        financial_output[
            column
        ] = pd.NA

    financial_output = financial_output[
        [
            "analytics_category",
            "metric",
            "category",
            "value",
            "exception_count",
            "total_difference",
            "average_difference",
            "maximum_difference",
            "high_severity_count",
            "medium_severity_count",
            "low_severity_count",
            "escalated_cases",
            "pending_review_cases",
            "ml_supported_cases",
            "ml_disagreement_cases",
            "escalation_rate_percent",
            "ml_support_rate_percent",
            "ml_disagreement_rate_percent",
            "review_rate_percent",
        ]
    ]

    escalation_output = escalation_analytics.copy()

    escalation_output[
        "metric"
    ] = "ESCALATION_BY_EXCEPTION"

    escalation_output[
        "category"
    ] = escalation_output[
        "exception_type"
    ]

    escalation_output[
        "value"
    ] = escalation_output[
        "escalated_cases"
    ]

    for column in [
        "total_difference",
        "average_difference",
        "maximum_difference",
        "high_severity_count",
        "medium_severity_count",
        "low_severity_count",
        "pending_review_cases",
        "ml_supported_cases",
        "ml_disagreement_cases",
        "ml_support_rate_percent",
        "ml_disagreement_rate_percent",
        "review_rate_percent",
    ]:

        escalation_output[
            column
        ] = pd.NA

    escalation_output = escalation_output[
        [
            "analytics_category",
            "metric",
            "category",
            "value",
            "total_cases",
            "total_difference",
            "average_difference",
            "maximum_difference",
            "high_severity_count",
            "medium_severity_count",
            "low_severity_count",
            "escalated_cases",
            "pending_review_cases",
            "ml_supported_cases",
            "ml_disagreement_cases",
            "escalation_rate_percent",
            "ml_support_rate_percent",
            "ml_disagreement_rate_percent",
            "review_rate_percent",
        ]
    ]

    # --------------------------------------------------------
    # Rename total_cases to exception_count so all tables have
    # the same schema.
    # --------------------------------------------------------

    if "total_cases" in escalation_output.columns:

        escalation_output = (
            escalation_output.rename(
                columns={
                    "total_cases":
                    "exception_count"
                }
            )
        )

    output = pd.concat(
        [
            exception_output,
            severity_output,
            financial_output,
            escalation_output,
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

        exception_analytics = (
            build_exception_analytics(df)
        )

        severity_analytics = (
            build_severity_analytics(df)
        )

        financial_summary = (
            build_financial_summary(df)
        )

        escalation_analytics = (
            build_escalation_analytics(df)
        )

        validate_analytics(
            exception_analytics,
            severity_analytics,
            financial_summary,
            escalation_analytics,
        )

        output = save_output(
            exception_analytics,
            severity_analytics,
            financial_summary,
            escalation_analytics,
        )

        print()
        print(
            "Exception analytics saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print(
            f"Total analytics records: {len(output)}"
        )

        print()
        print(
            "Top exception types:"
        )

        print(
            exception_analytics[
                [
                    "exception_type",
                    "exception_count",
                    "escalated_cases",
                    "total_difference",
                ]
            ]
            .head(10)
            .to_string(index=False)
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