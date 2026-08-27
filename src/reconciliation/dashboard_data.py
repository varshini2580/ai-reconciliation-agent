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
AUDIT_FILE = RECON_DIR / "audit_log.csv"
REVIEW_FILE = RECON_DIR / "review_queue.csv"

OUTPUT_FILE = RECON_DIR / "dashboard_data.csv"


# ============================================================
# LOAD INPUT FILES
# ============================================================

def load_inputs():

    print("=" * 70)
    print("        PHASE 7.1 — DASHBOARD DATA PREPARATION")
    print("=" * 70)

    files = {
        "reconciliation": RECONCILIATION_FILE,
        "explanation": EXPLANATION_FILE,
        "hybrid": HYBRID_FILE,
        "decision": DECISION_FILE,
        "action": ACTION_FILE,
        "escalation": ESCALATION_FILE,
        "audit": AUDIT_FILE,
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
    print("[OK] All dashboard source files loaded")

    return data


# ============================================================
# VALIDATE SOURCE FILES
# ============================================================

def validate_sources(data):

    print()
    print("========== SOURCE VALIDATION ==========")

    expected_counts = {
        "reconciliation": 1000,
        "explanation": 200,
        "hybrid": 1000,
        "decision": 200,
        "action": 200,
        "escalation": 170,
        "audit": 200,
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

    # --------------------------------------------------------
    # Transaction ID uniqueness
    # --------------------------------------------------------

    for name, df in data.items():

        if (
            "transaction_id" in df.columns
            and df["transaction_id"].duplicated().any()
        ):

            raise ValueError(
                f"{name}: duplicate transaction IDs"
            )

    print(
        "[OK] Source transaction IDs unique"
    )


# ============================================================
# BUILD DASHBOARD DATA
# ============================================================

def build_dashboard_data(data):

    print()
    print("========== BUILDING DASHBOARD DATA ==========")

    reconciliation = data["reconciliation"]
    explanation = data["explanation"]
    hybrid = data["hybrid"]
    decision = data["decision"]
    action = data["action"]
    escalation = data["escalation"]
    audit = data["audit"]
    review = data["review"]

    # --------------------------------------------------------
    # Start from all 1000 reconciliation transactions.
    # --------------------------------------------------------

    result = reconciliation.copy()

    # --------------------------------------------------------
    # Keep only useful transaction-level fields.
    #
    # We do not expose the full reconciliation source table
    # directly to the dashboard.
    # --------------------------------------------------------

    base_columns = [
        "transaction_id",
        "order_amount",
        "payment_total",
        "bank_total",
        "settlement_gross_total",
        "settlement_net_total",
        "settlement_fee_total",
        "settlement_tax_total",
        "settlement_adjustment_total",
        "settlement_refund_total",
        "settlement_chargeback_total",
        "status",
        "exception_type",
        "difference",
    ]

    missing_base = [
        column
        for column in base_columns
        if column not in result.columns
    ]

    if missing_base:

        raise ValueError(
            "Missing reconciliation columns: "
            + ", ".join(missing_base)
        )

    result = result[
        base_columns
    ].copy()

    # --------------------------------------------------------
    # Exception explanation data
    #
    # Only 200 transactions are exceptions, so use a LEFT JOIN
    # to preserve all 1000 transactions.
    # --------------------------------------------------------

    explanation_columns = [
        "transaction_id",
        "severity",
        "ai_explanation",
        "recommended_action",
        "explanation_source",
    ]

    result = result.merge(
        explanation[explanation_columns],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Hybrid ML data
    # --------------------------------------------------------

    hybrid_columns = [
        "transaction_id",
        "ml_status",
        "ml_exception_probability",
        "ml_predicted_exception_type",
        "ml_exception_type_probability",
        "hybrid_decision",
        "ml_agrees_with_rule_status",
        "ml_agrees_with_rule_exception_type",
    ]

    result = result.merge(
        hybrid[hybrid_columns],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Agent decision
    #
    # Only exception cases have agent decisions.
    # --------------------------------------------------------

    decision_columns = [
        "transaction_id",
        "agent_decision",
        "agent_decision_reason",
        "escalation_required",
    ]

    result = result.merge(
        decision[decision_columns],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Agent action
    # --------------------------------------------------------

    action_columns = [
        "transaction_id",
        "action_type",
        "action_status",
        "requires_human",
        "execution_mode",
    ]

    result = result.merge(
        action[action_columns],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Escalation
    #
    # Only 170 transactions have escalation cases.
    # --------------------------------------------------------

    escalation_columns = [
        "transaction_id",
        "escalation_priority",
        "escalation_status",
    ]

    result = result.merge(
        escalation[escalation_columns],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Audit information
    # --------------------------------------------------------

    audit_columns = [
        "transaction_id",
        "audit_id",
        "audit_event_status",
    ]

    result = result.merge(
        audit[audit_columns],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Review information
    # --------------------------------------------------------

    review_columns = [
        "transaction_id",
        "review_case_id",
        "review_priority",
        "queue_status",
        "review_status",
        "requires_human_review",
    ]

    result = result.merge(
        review[review_columns],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Dashboard-friendly derived fields
    # --------------------------------------------------------

    result["is_exception"] = (
        result["status"] == "EXCEPTION"
    ).astype(int)

    result["is_escalated"] = (
        result["escalation_required"]
        .fillna("NO")
        .astype(str)
        .str.upper()
        .eq("YES")
        .astype(int)
    )

    result["is_pending_review"] = (
        result["review_status"]
        .fillna("")
        .eq("PENDING")
        .astype(int)
    )

    result["ml_supported_exception"] = (
        result["hybrid_decision"]
        .fillna("")
        .eq("RULE_EXCEPTION_ML_SUPPORT")
        .astype(int)
    )

    result["ml_disagreement"] = (
        result["hybrid_decision"]
        .fillna("")
        .eq("RULE_EXCEPTION_ML_DISAGREEMENT")
        .astype(int)
    )

    result["ml_confirmed_rule_match"] = (
        result["hybrid_decision"]
        .fillna("")
        .eq("RULE_CONFIRMED_BY_ML")
        .astype(int)
    )

    # --------------------------------------------------------
    # Final dashboard columns
    # --------------------------------------------------------

    dashboard_columns = [
        "transaction_id",

        "order_amount",
        "payment_total",
        "bank_total",
        "settlement_gross_total",
        "settlement_net_total",
        "settlement_fee_total",
        "settlement_tax_total",
        "settlement_adjustment_total",
        "settlement_refund_total",
        "settlement_chargeback_total",

        "status",
        "exception_type",
        "severity",
        "difference",

        "ai_explanation",
        "recommended_action",
        "explanation_source",

        "ml_status",
        "ml_exception_probability",
        "ml_predicted_exception_type",
        "ml_exception_type_probability",
        "hybrid_decision",
        "ml_agrees_with_rule_status",
        "ml_agrees_with_rule_exception_type",

        "agent_decision",
        "agent_decision_reason",

        "action_type",
        "action_status",
        "requires_human",
        "execution_mode",

        "escalation_required",
        "escalation_priority",
        "escalation_status",

        "audit_id",
        "audit_event_status",

        "review_case_id",
        "review_priority",
        "queue_status",
        "review_status",
        "requires_human_review",

        "is_exception",
        "is_escalated",
        "is_pending_review",
        "ml_supported_exception",
        "ml_disagreement",
        "ml_confirmed_rule_match",
    ]

    result = result[
        dashboard_columns
    ]

    print(
        f"Dashboard records created: {len(result)}"
    )

    print(
        f"Dashboard fields: {len(result.columns)}"
    )

    return result


# ============================================================
# VALIDATE DASHBOARD DATA
# ============================================================

def validate_dashboard_data(df):

    print()
    print("=" * 70)
    print("        PHASE 7.1 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    if len(df) != 1000:

        raise ValueError(
            f"Expected 1000 dashboard records, "
            f"found {len(df)}"
        )

    print(
        "[OK] Dashboard record count: 1000"
    )

    # --------------------------------------------------------
    # Transaction IDs
    # --------------------------------------------------------

    if df[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected"
        )

    print(
        "[OK] Transaction IDs unique"
    )

    # --------------------------------------------------------
    # Exception count
    # --------------------------------------------------------

    exception_count = (
        df["is_exception"]
        .sum()
    )

    if exception_count != 200:

        raise ValueError(
            f"Expected 200 exceptions, "
            f"found {exception_count}"
        )

    print(
        "[OK] Exception count: 200"
    )

    # --------------------------------------------------------
    # Matched count
    # --------------------------------------------------------

    matched_count = (
        (df["is_exception"] == 0)
        .sum()
    )

    if matched_count != 800:

        raise ValueError(
            f"Expected 800 matched records, "
            f"found {matched_count}"
        )

    print(
        "[OK] Matched count: 800"
    )

    # --------------------------------------------------------
    # Escalations
    # --------------------------------------------------------

    escalation_count = (
        df["is_escalated"]
        .sum()
    )

    if escalation_count != 170:

        raise ValueError(
            f"Expected 170 escalations, "
            f"found {escalation_count}"
        )

    print(
        "[OK] Escalation count: 170"
    )

    # --------------------------------------------------------
    # Pending reviews
    # --------------------------------------------------------

    pending_count = (
        df["is_pending_review"]
        .sum()
    )

    if pending_count != 200:

        raise ValueError(
            f"Expected 200 pending reviews, "
            f"found {pending_count}"
        )

    print(
        "[OK] Pending review count: 200"
    )

    # --------------------------------------------------------
    # ML disagreement
    # --------------------------------------------------------

    disagreement_count = (
        df["ml_disagreement"]
        .sum()
    )

    if disagreement_count != 9:

        raise ValueError(
            f"Expected 9 ML disagreements, "
            f"found {disagreement_count}"
        )

    print(
        "[OK] ML disagreement count: 9"
    )

    # --------------------------------------------------------
    # No high-confidence ML exception overriding
    # deterministic match
    # --------------------------------------------------------

    matched_ml_exception = df[
        (df["is_exception"] == 0)
        & (
            df["ml_exception_probability"]
            .fillna(0)
            >= 0.80
        )
    ]

    if len(matched_ml_exception) != 0:

        raise ValueError(
            "Found high-confidence ML exceptions "
            "inside deterministic matches"
        )

    print(
        "[OK] No high-confidence ML override of rules"
    )

    # --------------------------------------------------------
    # Execution mode
    # --------------------------------------------------------

    execution_modes = (
        df["execution_mode"]
        .dropna()
        .astype(str)
        .unique()
    )

    if set(execution_modes) != {"SIMULATED"}:

        raise ValueError(
            "Unexpected execution mode"
        )

    print(
        "[OK] Execution mode = SIMULATED"
    )

    # --------------------------------------------------------
    # Critical missing values
    # --------------------------------------------------------

    critical_columns = [
        "transaction_id",
        "status",
        "is_exception",
        "is_escalated",
    ]

    missing = (
        df[critical_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing:

        raise ValueError(
            f"Critical dashboard values missing: {missing}"
        )

    print(
        "[OK] No missing critical dashboard values"
    )

    # --------------------------------------------------------
    # Summary distributions
    # --------------------------------------------------------

    print()
    print(
        "Status distribution:"
    )

    print(
        df["status"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Severity distribution:"
    )

    print(
        df["severity"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print(
        "Hybrid decision distribution:"
    )

    print(
        df["hybrid_decision"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print(
        "Agent decision distribution:"
    )

    print(
        df["agent_decision"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print(
        "Escalation distribution:"
    )

    print(
        df["escalation_required"]
        .value_counts(dropna=False)
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

        dashboard = build_dashboard_data(
            data
        )

        validate_dashboard_data(
            dashboard
        )

        dashboard.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Dashboard data saved to:"
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