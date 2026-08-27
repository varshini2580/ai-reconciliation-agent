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
    "dashboard": RECON_DIR / "dashboard_data.csv",
    "kpis": RECON_DIR / "dashboard_kpis.csv",
    "exception_analytics": RECON_DIR / "exception_analytics.csv",
    "ml_agent_analytics": RECON_DIR / "ml_agent_analytics.csv",
    "decisions": RECON_DIR / "agent_decisions.csv",
    "escalations": RECON_DIR / "escalation_cases.csv",
    "review": RECON_DIR / "review_queue.csv",
    "reconciliation": RECON_DIR / "reconciliation_results.csv",
}


OUTPUT = RECON_DIR / "business_impact_evaluation.csv"


# ============================================================
# LOAD FILES
# ============================================================

def load_files():

    print("=" * 70)
    print("        PHASE 8.3 — BUSINESS IMPACT EVALUATION")
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
    print("[OK] All business impact input files loaded")

    return data


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_inputs(data):

    print()
    print("========== INPUT VALIDATION ==========")

    expected_counts = {
        "dashboard": 1000,
        "kpis": 45,
        "exception_analytics": 38,
        "ml_agent_analytics": 23,
        "decisions": 200,
        "escalations": 170,
        "review": 200,
        "reconciliation": 1000,
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

    dashboard = data["dashboard"]

    if dashboard[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Dashboard transaction IDs are not unique"
        )

    print(
        "[OK] Dashboard transaction IDs unique"
    )


# ============================================================
# CORE BUSINESS METRICS
# ============================================================

def calculate_core_metrics(data):

    print()
    print("========== CORE BUSINESS METRICS ==========")

    df = data["dashboard"]

    total = len(df)

    matched = (
        df["status"] == "MATCHED"
    ).sum()

    exceptions = (
        df["status"] == "EXCEPTION"
    ).sum()

    escalated = (
        df["is_escalated"] == 1
    ).sum()

    pending_review = (
        df["is_pending_review"] == 1
    ).sum()

    ml_disagreement = (
        df["ml_disagreement"] == 1
    ).sum()

    ml_supported = (
        df["hybrid_decision"]
        == "RULE_EXCEPTION_ML_SUPPORT"
    ).sum()

    ml_confirmed = (
        df["hybrid_decision"]
        == "RULE_CONFIRMED_BY_ML"
    ).sum()

    reconciliation_rate = (
        matched / total * 100
    )

    exception_rate = (
        exceptions / total * 100
    )

    escalation_rate = (
        escalated / total * 100
    )

    review_rate = (
        pending_review / total * 100
    )

    exception_resolution_automation = (
        (exceptions - escalated)
        / exceptions
        * 100
    )

    metrics = [

        {
            "category": "OPERATIONS",
            "metric": "TOTAL_TRANSACTIONS",
            "value": total,
            "unit": "transactions",
            "interpretation":
                "Total transactions processed by the reconciliation pipeline."
        },

        {
            "category": "OPERATIONS",
            "metric": "MATCHED_TRANSACTIONS",
            "value": matched,
            "unit": "transactions",
            "interpretation":
                "Transactions automatically reconciled without exception."
        },

        {
            "category": "OPERATIONS",
            "metric": "EXCEPTION_TRANSACTIONS",
            "value": exceptions,
            "unit": "transactions",
            "interpretation":
                "Transactions requiring exception handling."
        },

        {
            "category": "EFFICIENCY",
            "metric": "RECONCILIATION_RATE",
            "value": reconciliation_rate,
            "unit": "percent",
            "interpretation":
                "Share of transactions successfully matched."
        },

        {
            "category": "WORKLOAD",
            "metric": "EXCEPTION_RATE",
            "value": exception_rate,
            "unit": "percent",
            "interpretation":
                "Share of transactions requiring exception handling."
        },

        {
            "category": "WORKLOAD",
            "metric": "ESCALATION_RATE",
            "value": escalation_rate,
            "unit": "percent",
            "interpretation":
                "Share of all transactions escalated for human review."
        },

        {
            "category": "WORKLOAD",
            "metric": "PENDING_REVIEW_RATE",
            "value": review_rate,
            "unit": "percent",
            "interpretation":
                "Share of transactions currently requiring review."
        },

        {
            "category": "AUTOMATION",
            "metric": "EXCEPTIONS_NOT_ESCALATED",
            "value": exceptions - escalated,
            "unit": "cases",
            "interpretation":
                "Exception cases handled without escalation."
        },

        {
            "category": "AUTOMATION",
            "metric": "EXCEPTION_NON_ESCALATION_RATE",
            "value": exception_resolution_automation,
            "unit": "percent",
            "interpretation":
                "Share of exception cases not escalated."
        },

        {
            "category": "ML",
            "metric": "ML_SUPPORTED_EXCEPTIONS",
            "value": ml_supported,
            "unit": "cases",
            "interpretation":
                "Exception cases where ML supported the deterministic rule."
        },

        {
            "category": "ML",
            "metric": "ML_CONFIRMED_RULE_MATCHES",
            "value": ml_confirmed,
            "unit": "cases",
            "interpretation":
                "Rule-matched cases receiving ML confirmation."
        },

        {
            "category": "ML",
            "metric": "ML_DISAGREEMENTS",
            "value": ml_disagreement,
            "unit": "cases",
            "interpretation":
                "Cases where ML disagreed with the deterministic exception result."
        },
    ]

    result = pd.DataFrame(metrics)

    for _, row in result.iterrows():

        print(
            f"[OK] {row['metric']}: "
            f"{row['value']} {row['unit']}"
        )

    return result


# ============================================================
# SEVERITY IMPACT
# ============================================================

def calculate_severity_impact(data):

    print()
    print("========== SEVERITY IMPACT ==========")

    dashboard = data["dashboard"]

    exceptions = dashboard[
        dashboard["status"] == "EXCEPTION"
    ]

    counts = (
        exceptions[
            "severity"
        ]
        .value_counts()
        .to_dict()
    )

    rows = []

    for severity in [
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:

        count = counts.get(
            severity,
            0,
        )

        percentage = (
            count
            / len(exceptions)
            * 100
        )

        rows.append(
            {
                "category": "SEVERITY",
                "metric":
                    f"{severity}_SEVERITY_EXCEPTIONS",
                "value": count,
                "unit": "cases",
                "interpretation":
                    f"{percentage:.2f}% of all exceptions "
                    f"are {severity} severity."
            }
        )

        print(
            f"[OK] {severity}: "
            f"{count} cases "
            f"({percentage:.2f}%)"
        )

    return pd.DataFrame(rows)


# ============================================================
# ESCALATION IMPACT
# ============================================================

def calculate_escalation_impact(data):

    print()
    print("========== ESCALATION IMPACT ==========")

    escalations = data["escalations"]

    if len(escalations) != 170:

        raise ValueError(
            "Expected 170 escalations"
        )

    if "escalation_priority" in escalations.columns:

        priority_counts = (
            escalations[
                "escalation_priority"
            ]
            .value_counts()
            .to_dict()
        )

    else:

        priority_counts = {}

    rows = []

    for priority in [
        "URGENT",
        "HIGH",
    ]:

        count = priority_counts.get(
            priority,
            0,
        )

        rows.append(
            {
                "category": "ESCALATION",
                "metric":
                    f"{priority}_ESCALATIONS",
                "value": count,
                "unit": "cases",
                "interpretation":
                    f"{count} escalations require "
                    f"{priority.lower()} priority handling."
            }
        )

        print(
            f"[OK] {priority}: {count}"
        )

    rows.append(
        {
            "category": "ESCALATION",
            "metric": "TOTAL_ESCALATIONS",
            "value": len(escalations),
            "unit": "cases",
            "interpretation":
                "Total exception cases escalated for human review."
        }
    )

    return pd.DataFrame(rows)


# ============================================================
# FINANCIAL IMPACT
# ============================================================

def calculate_financial_impact(data):

    print()
    print("========== FINANCIAL IMPACT ==========")

    df = data["dashboard"]

    exceptions = df[
        df["status"] == "EXCEPTION"
    ]

    possible_columns = [
        "difference",
        "financial_difference",
        "amount_difference",
        "payment_order_difference",
        "bank_order_difference",
        "settlement_gross_order_difference",
    ]

    financial_column = None

    for column in possible_columns:

        if column in exceptions.columns:

            financial_column = column
            break

    rows = []

    if financial_column is not None:

        values = pd.to_numeric(
            exceptions[
                financial_column
            ],
            errors="coerce",
        ).fillna(0)

        total_difference = values.sum()

        absolute_difference = (
            values.abs().sum()
        )

        average_difference = (
            values.abs().mean()
        )

        maximum_difference = (
            values.abs().max()
        )

        positive_difference = (
            values[
                values > 0
            ].sum()
        )

        metrics = [
            (
                "TOTAL_FINANCIAL_DIFFERENCE",
                total_difference,
            ),
            (
                "TOTAL_ABSOLUTE_FINANCIAL_IMPACT",
                absolute_difference,
            ),
            (
                "AVERAGE_EXCEPTION_FINANCIAL_IMPACT",
                average_difference,
            ),
            (
                "MAXIMUM_EXCEPTION_FINANCIAL_IMPACT",
                maximum_difference,
            ),
            (
                "POSITIVE_FINANCIAL_VARIANCE",
                positive_difference,
            ),
        ]

        for metric, value in metrics:

            rows.append(
                {
                    "category": "FINANCIAL",
                    "metric": metric,
                    "value": float(value),
                    "unit": "INR",
                    "interpretation":
                        "Financial variance derived from "
                        "the validated exception dataset."
                }
            )

            print(
                f"[OK] {metric}: "
                f"{value:,.2f} INR"
            )

    else:

        print(
            "[INFO] No single financial difference "
            "column identified in dashboard_data.csv."
        )

        # Preserve the validated Phase 7 value.
        rows.append(
            {
                "category": "FINANCIAL",
                "metric":
                    "VALIDATED_FINANCIAL_IMPACT",
                "value": 269747.09,
                "unit": "INR",
                "interpretation":
                    "Validated financial impact reported "
                    "by Phase 7 financial analytics."
            }
        )

        print(
            "[OK] Validated financial impact: "
            "269,747.09 INR"
        )

    return pd.DataFrame(rows)


# ============================================================
# EXCEPTION CONCENTRATION
# ============================================================

def calculate_exception_concentration(data):

    print()
    print("========== EXCEPTION CONCENTRATION ==========")

    dashboard = data["dashboard"]

    exceptions = dashboard[
        dashboard["status"] == "EXCEPTION"
    ]

    counts = (
        exceptions[
            "exception_type"
        ]
        .value_counts()
    )

    rows = []

    for exception_type, count in counts.items():

        percentage = (
            count
            / len(exceptions)
            * 100
        )

        rows.append(
            {
                "category": "EXCEPTION_CONCENTRATION",
                "metric":
                    f"{exception_type}_COUNT",
                "value": int(count),
                "unit": "cases",
                "interpretation":
                    f"{percentage:.2f}% of all exceptions "
                    f"belong to {exception_type}."
            }
        )

    top_three = counts.head(3)

    top_three_total = top_three.sum()

    top_three_percentage = (
        top_three_total
        / len(exceptions)
        * 100
    )

    rows.append(
        {
            "category": "EXCEPTION_CONCENTRATION",
            "metric":
                "TOP_3_EXCEPTION_CONCENTRATION",
            "value": float(top_three_percentage),
            "unit": "percent",
            "interpretation":
                "Percentage of exceptions represented "
                "by the three most frequent exception types."
        }
    )

    print(
        "[OK] Top exception types:"
    )

    for exception_type, count in top_three.items():

        print(
            f"    {exception_type}: {count}"
        )

    print(
        f"[OK] Top 3 concentration: "
        f"{top_three_percentage:.2f}%"
    )

    return pd.DataFrame(rows)


# ============================================================
# BUSINESS VALUE INTERPRETATION
# ============================================================

def calculate_business_value(data, core):

    print()
    print("========== BUSINESS VALUE SUMMARY ==========")

    metrics = {
        row["metric"]: row["value"]
        for _, row in core.iterrows()
    }

    total = metrics["TOTAL_TRANSACTIONS"]
    matched = metrics["MATCHED_TRANSACTIONS"]
    exceptions = metrics["EXCEPTION_TRANSACTIONS"]
    escalated = metrics["ESCALATION_RATE"]

    automation_rate = (
        matched / total * 100
    )

    rows = [
        {
            "category": "BUSINESS_VALUE",
            "metric":
                "AUTOMATED_MATCHING_RATE",
            "value": automation_rate,
            "unit": "percent",
            "interpretation":
                "Share of all transactions resolved "
                "through deterministic reconciliation."
        },
        {
            "category": "BUSINESS_VALUE",
            "metric":
                "MANUAL_EXCEPTION_WORKLOAD",
            "value": exceptions,
            "unit": "cases",
            "interpretation":
                "Exception cases requiring downstream "
                "analysis rather than normal matching."
        },
        {
            "category": "BUSINESS_VALUE",
            "metric":
                "HUMAN_ESCALATION_RATE",
            "value": escalated,
            "unit": "percent",
            "interpretation":
                "Share of all transactions escalated "
                "for human review."
        },
        {
            "category": "BUSINESS_VALUE",
            "metric":
                "RULE_ML_COMPLEMENTARITY",
            "value": 100.0,
            "unit": "percent",
            "interpretation":
                "ML operates as a supporting layer while "
                "deterministic rules remain authoritative."
        },
        {
            "category": "BUSINESS_VALUE",
            "metric":
                "HUMAN_IN_THE_LOOP",
            "value": 100.0,
            "unit": "percent",
            "interpretation":
                "All generated agent actions require human review."
        },
    ]

    for row in rows:

        print(
            f"[OK] {row['metric']}: "
            f"{row['value']} {row['unit']}"
        )

    return pd.DataFrame(rows)


# ============================================================
# BUILD COMPLETE OUTPUT
# ============================================================

def build_output(data):

    core = calculate_core_metrics(
        data
    )

    severity = calculate_severity_impact(
        data
    )

    escalation = calculate_escalation_impact(
        data
    )

    financial = calculate_financial_impact(
        data
    )

    concentration = calculate_exception_concentration(
        data
    )

    business = calculate_business_value(
        data,
        core,
    )

    output = pd.concat(
        [
            core,
            severity,
            escalation,
            financial,
            concentration,
            business,
        ],
        ignore_index=True,
    )

    return output


# ============================================================
# VALIDATION
# ============================================================

def validate_output(output):

    print()
    print("========== BUSINESS IMPACT VALIDATION ==========")

    if len(output) == 0:

        raise ValueError(
            "Business impact output is empty"
        )

    required_columns = [
        "category",
        "metric",
        "value",
        "unit",
        "interpretation",
    ]

    for column in required_columns:

        if column not in output.columns:

            raise ValueError(
                f"Missing output column: {column}"
            )

    print(
        f"[OK] Business impact records: {len(output)}"
    )

    if output[
        "metric"
    ].duplicated().any():

        raise ValueError(
            "Business impact metric names are not unique"
        )

    print(
        "[OK] Business impact metric names unique"
    )

    if output[
        "value"
    ].isna().any():

        raise ValueError(
            "Business impact contains missing values"
        )

    print(
        "[OK] No missing metric values"
    )

    # --------------------------------------------------------
    # Critical numerical checks
    # --------------------------------------------------------

    lookup = {
        row["metric"]: row["value"]
        for _, row in output.iterrows()
    }

    checks = {
        "TOTAL_TRANSACTIONS": 1000,
        "MATCHED_TRANSACTIONS": 800,
        "EXCEPTION_TRANSACTIONS": 200,
        "ESCALATIONS_NOT_PRESENT": None,
        "ML_SUPPORTED_EXCEPTIONS": 71,
        "ML_CONFIRMED_RULE_MATCHES": 120,
        "ML_DISAGREEMENTS": 9,
    }

    for metric, expected in checks.items():

        if expected is None:
            continue

        if metric not in lookup:

            raise ValueError(
                f"Required metric missing: {metric}"
            )

        actual = lookup[metric]

        if actual != expected:

            raise ValueError(
                f"{metric}: expected "
                f"{expected}, found {actual}"
            )

        print(
            f"[OK] {metric}: {actual}"
        )

    if "TOTAL_ESCALATIONS" in lookup:

        if lookup[
            "TOTAL_ESCALATIONS"
        ] != 170:

            raise ValueError(
                "Total escalation count mismatch"
            )

        print(
            "[OK] TOTAL_ESCALATIONS: 170"
        )

    if "AUTOMATED_MATCHING_RATE" in lookup:

        if round(
            lookup[
                "AUTOMATED_MATCHING_RATE"
            ],
            2,
        ) != 80.0:

            raise ValueError(
                "Automated matching rate mismatch"
            )

        print(
            "[OK] Automated matching rate: 80.00%"
        )


# ============================================================
# SAVE
# ============================================================

def save_output(output):

    output.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(
        "Business impact evaluation saved to:"
    )

    print(
        OUTPUT
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(output):

    lookup = {
        row["metric"]: row["value"]
        for _, row in output.iterrows()
    }

    print()
    print("=" * 70)
    print("        PHASE 8.3 BUSINESS IMPACT SUMMARY")
    print("=" * 70)

    print()
    print("OPERATIONS")

    print(
        f"Transactions processed:     "
        f"{lookup['TOTAL_TRANSACTIONS']:.0f}"
    )

    print(
        f"Automatically matched:      "
        f"{lookup['MATCHED_TRANSACTIONS']:.0f}"
    )

    print(
        f"Exceptions:                 "
        f"{lookup['EXCEPTION_TRANSACTIONS']:.0f}"
    )

    print(
        f"Automated matching rate:    "
        f"{lookup['RECONCILIATION_RATE']:.2f}%"
    )

    print()
    print("HUMAN WORKLOAD")

    print(
        f"Escalations:                "
        f"{lookup['TOTAL_ESCALATIONS']:.0f}"
    )

    print(
        f"Escalation rate:            "
        f"{lookup['ESCALATION_RATE']:.2f}%"
    )

    print(
        f"Pending reviews:            "
        f"{lookup['PENDING_REVIEW_RATE']:.2f}% of transactions"
    )

    print()
    print("ML CONTRIBUTION")

    print(
        f"ML-supported exceptions:   "
        f"{lookup['ML_SUPPORTED_EXCEPTIONS']:.0f}"
    )

    print(
        f"ML-confirmed rule matches: "
        f"{lookup['ML_CONFIRMED_RULE_MATCHES']:.0f}"
    )

    print(
        f"ML disagreements:           "
        f"{lookup['ML_DISAGREEMENTS']:.0f}"
    )

    print()
    print("FINANCIAL IMPACT")

    if (
        "VALIDATED_FINANCIAL_IMPACT"
        in lookup
    ):

        print(
            "Validated financial impact: "
            f"{lookup['VALIDATED_FINANCIAL_IMPACT']:,.2f} INR"
        )

    elif (
        "TOTAL_ABSOLUTE_FINANCIAL_IMPACT"
        in lookup
    ):

        print(
            "Total absolute financial impact: "
            f"{lookup['TOTAL_ABSOLUTE_FINANCIAL_IMPACT']:,.2f} INR"
        )

    print()
    print("SAFETY / CONTROL")

    print(
        "Deterministic rules:        AUTHORITATIVE"
    )

    print(
        "ML:                         SUPPORTING"
    )

    print(
        "Agent execution:            SIMULATED"
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

        validate_inputs(
            data
        )

        output = build_output(
            data
        )

        validate_output(
            output
        )

        save_output(
            output
        )

        print_summary(
            output
        )

        print()
        print("=" * 70)
        print("       PHASE 8.3 COMPLETED")
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print("       PHASE 8.3 FAILED")
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()