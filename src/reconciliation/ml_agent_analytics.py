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

OUTPUT_FILE = RECON_DIR / "ml_agent_analytics.csv"


# ============================================================
# LOAD
# ============================================================

def load_dashboard_data():

    print("=" * 70)
    print("        PHASE 7.4 — ML + AGENT ANALYTICS")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dashboard data not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Dashboard records: {len(df)}")

    return df


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(df):

    print()
    print("========== INPUT VALIDATION ==========")

    required_columns = [
        "transaction_id",
        "status",
        "exception_type",
        "is_exception",

        "ml_status",
        "ml_exception_probability",
        "ml_predicted_exception_type",
        "ml_exception_type_probability",

        "hybrid_decision",
        "ml_agrees_with_rule_status",
        "ml_agrees_with_rule_exception_type",

        "ml_supported_exception",
        "ml_disagreement",
        "ml_confirmed_rule_match",

        "agent_decision",
        "agent_decision_reason",

        "action_type",
        "action_status",
        "requires_human",
        "execution_mode",

        "escalation_required",
        "escalation_priority",
        "escalation_status",

        "review_status",
        "requires_human_review",
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

    print("[OK] Dashboard schema valid")

    if len(df) != 1000:
        raise ValueError(
            f"Expected 1000 records, found {len(df)}"
        )

    print("[OK] 1000 dashboard records loaded")

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs detected"
        )

    print("[OK] Transaction IDs unique")


# ============================================================
# ML SUMMARY
# ============================================================

def build_ml_summary(df):

    print()
    print("========== BUILDING ML SUMMARY ==========")

    exceptions = df[
        df["is_exception"] == 1
    ].copy()

    total_exceptions = len(exceptions)

    rule_confirmed = (
        exceptions["hybrid_decision"]
        == "RULE_CONFIRMED_BY_ML"
    ).sum()

    ml_supported = (
        exceptions["hybrid_decision"]
        == "RULE_EXCEPTION_ML_SUPPORT"
    ).sum()

    disagreements = (
        exceptions["hybrid_decision"]
        == "RULE_EXCEPTION_ML_DISAGREEMENT"
    ).sum()

    ml_status_agreement = (
        exceptions[
            "ml_agrees_with_rule_status"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    ml_exception_type_agreement = (
        exceptions[
            "ml_agrees_with_rule_exception_type"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    average_exception_probability = (
        pd.to_numeric(
            exceptions[
                "ml_exception_probability"
            ],
            errors="coerce",
        )
        .mean()
    )

    average_exception_type_probability = (
        pd.to_numeric(
            exceptions[
                "ml_exception_type_probability"
            ],
            errors="coerce",
        )
        .mean()
    )

    result = pd.DataFrame(
        [
            {
                "analytics_category": "ML_SUMMARY",
                "metric": "TOTAL_EXCEPTION_CASES",
                "value": total_exceptions,
            },
            {
                "analytics_category": "ML_SUMMARY",
                "metric": "RULE_CONFIRMED_BY_ML",
                "value": rule_confirmed,
            },
            {
                "analytics_category": "ML_SUMMARY",
                "metric": "RULE_EXCEPTION_ML_SUPPORT",
                "value": ml_supported,
            },
            {
                "analytics_category": "ML_SUMMARY",
                "metric": "RULE_EXCEPTION_ML_DISAGREEMENT",
                "value": disagreements,
            },
            {
                "analytics_category": "ML_SUMMARY",
                "metric": "ML_STATUS_AGREEMENT_CASES",
                "value": ml_status_agreement,
            },
            {
                "analytics_category": "ML_SUMMARY",
                "metric": "ML_EXCEPTION_TYPE_AGREEMENT_CASES",
                "value": ml_exception_type_agreement,
            },
            {
                "analytics_category": "ML_SUMMARY",
                "metric": "AVERAGE_EXCEPTION_PROBABILITY",
                "value": round(
                    average_exception_probability,
                    4,
                ),
            },
            {
                "analytics_category": "ML_SUMMARY",
                "metric": "AVERAGE_EXCEPTION_TYPE_PROBABILITY",
                "value": round(
                    average_exception_type_probability,
                    4,
                ),
            },
        ]
    )

    print(
        f"ML summary metrics created: {len(result)}"
    )

    return result


# ============================================================
# HYBRID DECISION ANALYTICS
# ============================================================

def build_hybrid_analytics(df):

    print()
    print("========== BUILDING HYBRID ANALYTICS ==========")

    result = (
        df.groupby(
            "hybrid_decision"
        )
        .agg(
            case_count=(
                "transaction_id",
                "count",
            ),
            exception_count=(
                "is_exception",
                "sum",
            ),
        )
        .reset_index()
    )

    result[
        "analytics_category"
    ] = "HYBRID_DECISION"

    result[
        "percentage"
    ] = (
        result["case_count"]
        / len(df)
        * 100
    ).round(2)

    result = result[
        [
            "analytics_category",
            "hybrid_decision",
            "case_count",
            "exception_count",
            "percentage",
        ]
    ]

    print(
        f"Hybrid decision categories: {len(result)}"
    )

    return result


# ============================================================
# AGENT DECISION ANALYTICS
# ============================================================

def build_agent_analytics(df):

    print()
    print("========== BUILDING AGENT ANALYTICS ==========")

    exceptions = df[
        df["is_exception"] == 1
    ].copy()

    result = (
        exceptions.groupby(
            "agent_decision"
        )
        .agg(
            case_count=(
                "transaction_id",
                "count",
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
        "analytics_category"
    ] = "AGENT_DECISION"

    result[
        "escalation_rate_percent"
    ] = (
        result["escalated_cases"]
        / result["case_count"]
        * 100
    ).round(2)

    result[
        "review_rate_percent"
    ] = (
        result["pending_review_cases"]
        / result["case_count"]
        * 100
    ).round(2)

    result = result[
        [
            "analytics_category",
            "agent_decision",
            "case_count",
            "escalated_cases",
            "pending_review_cases",
            "escalation_rate_percent",
            "review_rate_percent",
        ]
    ]

    print(
        f"Agent decision categories: {len(result)}"
    )

    return result


# ============================================================
# ACTION ANALYTICS
# ============================================================

def build_action_analytics(df):

    print()
    print("========== BUILDING ACTION ANALYTICS ==========")

    exceptions = df[
        df["is_exception"] == 1
    ].copy()

    result = (
        exceptions.groupby(
            "action_type"
        )
        .agg(
            case_count=(
                "transaction_id",
                "count",
            ),
            pending_cases=(
                "action_status",
                lambda x: (
                    x == "PENDING_REVIEW"
                ).sum(),
            ),
            review_required_cases=(
                "action_status",
                lambda x: (
                    x == "REVIEW_REQUIRED"
                ).sum(),
            ),
        )
        .reset_index()
    )

    result[
        "analytics_category"
    ] = "ACTION_TYPE"

    result = result[
        [
            "analytics_category",
            "action_type",
            "case_count",
            "pending_cases",
            "review_required_cases",
        ]
    ]

    print(
        f"Action categories: {len(result)}"
    )

    return result


# ============================================================
# ESCALATION ANALYTICS
# ============================================================

def build_escalation_analytics(df):

    print()
    print("========== BUILDING ML / AGENT ESCALATION ANALYTICS ==========")

    exceptions = df[
        df["is_exception"] == 1
    ].copy()

    result = (
        exceptions.groupby(
            "hybrid_decision"
        )
        .agg(
            case_count=(
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
        "analytics_category"
    ] = "ESCALATION_BY_HYBRID_DECISION"

    result[
        "escalation_rate_percent"
    ] = (
        result["escalated_cases"]
        / result["case_count"]
        * 100
    ).round(2)

    result = result[
        [
            "analytics_category",
            "hybrid_decision",
            "case_count",
            "escalated_cases",
            "escalation_rate_percent",
        ]
    ]

    print(
        f"Escalation categories: {len(result)}"
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_analytics(
    ml_summary,
    hybrid_analytics,
    agent_analytics,
    action_analytics,
    escalation_analytics,
):

    print()
    print("=" * 70)
    print("        PHASE 7.4 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # ML summary
    # --------------------------------------------------------

    if len(ml_summary) != 8:
        raise ValueError(
            f"Expected 8 ML summary metrics, "
            f"found {len(ml_summary)}"
        )

    print("[OK] ML summary metrics: 8")

    values = dict(
        zip(
            ml_summary["metric"],
            ml_summary["value"],
        )
    )

    expected = {
        "TOTAL_EXCEPTION_CASES": 200,
        "RULE_CONFIRMED_BY_ML": 120,
        "RULE_EXCEPTION_ML_SUPPORT": 71,
        "RULE_EXCEPTION_ML_DISAGREEMENT": 9,
        "ML_STATUS_AGREEMENT_CASES": 191,
        "ML_EXCEPTION_TYPE_AGREEMENT_CASES": 192,
    }

    for metric, expected_value in expected.items():

        actual = values[metric]

        if actual != expected_value:
            raise ValueError(
                f"{metric}: expected "
                f"{expected_value}, found {actual}"
            )

        print(
            f"[OK] {metric}: {actual}"
        )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    if len(hybrid_analytics) != 4:
        raise ValueError(
            "Expected 4 hybrid decision categories"
        )

    hybrid_total = (
        hybrid_analytics[
            "case_count"
        ].sum()
    )

    if hybrid_total != 1000:
        raise ValueError(
            f"Hybrid analytics total should be 1000, "
            f"found {hybrid_total}"
        )

    print(
        "[OK] Hybrid decision analytics total: 1000"
    )

    # --------------------------------------------------------
    # Agent decisions
    # --------------------------------------------------------

    if len(agent_analytics) != 4:
        raise ValueError(
            "Expected 4 agent decision categories"
        )

    agent_total = (
        agent_analytics[
            "case_count"
        ].sum()
    )

    if agent_total != 200:
        raise ValueError(
            f"Agent analytics total should be 200, "
            f"found {agent_total}"
        )

    print(
        "[OK] Agent decision analytics total: 200"
    )

    # --------------------------------------------------------
    # Agent actions
    # --------------------------------------------------------

    if len(action_analytics) != 4:
        raise ValueError(
            "Expected 4 action categories"
        )

    action_total = (
        action_analytics[
            "case_count"
        ].sum()
    )

    if action_total != 200:
        raise ValueError(
            f"Action analytics total should be 200, "
            f"found {action_total}"
        )

    print(
        "[OK] Action analytics total: 200"
    )

    # --------------------------------------------------------
    # Escalation
    # --------------------------------------------------------

    escalation_total = (
        escalation_analytics[
            "escalated_cases"
        ].sum()
    )

    if escalation_total != 170:
        raise ValueError(
            f"Escalation analytics total should be 170, "
            f"found {escalation_total}"
        )

    print(
        "[OK] Escalation analytics total: 170"
    )

    # --------------------------------------------------------
    # Critical ML values
    # --------------------------------------------------------

    if values[
        "RULE_EXCEPTION_ML_SUPPORT"
    ] != 71:
        raise ValueError(
            "ML support count mismatch"
        )

    if values[
        "RULE_EXCEPTION_ML_DISAGREEMENT"
    ] != 9:
        raise ValueError(
            "ML disagreement count mismatch"
        )

    print(
        "[OK] ML support/disagreement counts preserved"
    )


# ============================================================
# SAVE
# ============================================================

def save_output(
    ml_summary,
    hybrid_analytics,
    agent_analytics,
    action_analytics,
    escalation_analytics,
):

    # --------------------------------------------------------
    # Save separate analytics tables into one CSV.
    # --------------------------------------------------------

    rows = []

    # --------------------------------------------------------
    # ML summary
    # --------------------------------------------------------

    for _, row in ml_summary.iterrows():

        rows.append(
            {
                "analytics_category":
                    "ML_SUMMARY",

                "metric":
                    row["metric"],

                "category":
                    pd.NA,

                "case_count":
                    row["value"],

                "secondary_count":
                    pd.NA,

                "percentage":
                    pd.NA,
            }
        )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    for _, row in hybrid_analytics.iterrows():

        rows.append(
            {
                "analytics_category":
                    "HYBRID_DECISION",

                "metric":
                    pd.NA,

                "category":
                    row["hybrid_decision"],

                "case_count":
                    row["case_count"],

                "secondary_count":
                    row["exception_count"],

                "percentage":
                    row["percentage"],
            }
        )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    for _, row in agent_analytics.iterrows():

        rows.append(
            {
                "analytics_category":
                    "AGENT_DECISION",

                "metric":
                    pd.NA,

                "category":
                    row["agent_decision"],

                "case_count":
                    row["case_count"],

                "secondary_count":
                    row["escalated_cases"],

                "percentage":
                    row["escalation_rate_percent"],
            }
        )

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    for _, row in action_analytics.iterrows():

        rows.append(
            {
                "analytics_category":
                    "ACTION_TYPE",

                "metric":
                    pd.NA,

                "category":
                    row["action_type"],

                "case_count":
                    row["case_count"],

                "secondary_count":
                    row["pending_cases"],

                "percentage":
                    pd.NA,
            }
        )

    # --------------------------------------------------------
    # Escalation
    # --------------------------------------------------------

    for _, row in escalation_analytics.iterrows():

        rows.append(
            {
                "analytics_category":
                    "ESCALATION_BY_HYBRID_DECISION",

                "metric":
                    pd.NA,

                "category":
                    row["hybrid_decision"],

                "case_count":
                    row["case_count"],

                "secondary_count":
                    row["escalated_cases"],

                "percentage":
                    row["escalation_rate_percent"],
            }
        )

    output = pd.DataFrame(
        rows,
        columns=[
            "analytics_category",
            "metric",
            "category",
            "case_count",
            "secondary_count",
            "percentage",
        ],
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

        ml_summary = build_ml_summary(
            df
        )

        hybrid_analytics = build_hybrid_analytics(
            df
        )

        agent_analytics = build_agent_analytics(
            df
        )

        action_analytics = build_action_analytics(
            df
        )

        escalation_analytics = (
            build_escalation_analytics(df)
        )

        validate_analytics(
            ml_summary,
            hybrid_analytics,
            agent_analytics,
            action_analytics,
            escalation_analytics,
        )

        output = save_output(
            ml_summary,
            hybrid_analytics,
            agent_analytics,
            action_analytics,
            escalation_analytics,
        )

        print()
        print(
            "ML + Agent analytics saved to:"
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
            "Hybrid decision distribution:"
        )

        print(
            hybrid_analytics[
                [
                    "hybrid_decision",
                    "case_count",
                    "percentage",
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "Agent decision distribution:"
        )

        print(
            agent_analytics[
                [
                    "agent_decision",
                    "case_count",
                    "escalated_cases",
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 7.4 COMPLETED"
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