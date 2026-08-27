from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
RECONCILIATION_DIR = DATA_DIR / "reconciliation"

RECONCILIATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SEVERITY
# ============================================================

SEVERITY = {

    "MISSING_PAYMENT": "HIGH",
    "MISSING_SETTLEMENT": "HIGH",
    "AMOUNT_MISMATCH": "HIGH",
    "FAILED_PAYMENT": "HIGH",
    "REFUND": "HIGH",
    "CHARGEBACK": "HIGH",

    "PARTIAL_SETTLEMENT": "MEDIUM",
    "DUPLICATE_PAYMENT": "MEDIUM",
    "DUPLICATE_SETTLEMENT": "MEDIUM",
    "MULTIPLE_PAYMENTS": "MEDIUM",
    "WRONG_TRANSACTION_REFERENCE": "MEDIUM",
    "UNKNOWN_ADJUSTMENT": "MEDIUM",
    "INCORRECT_FEE": "MEDIUM",

    "DATE_MISMATCH": "LOW",
    "SETTLEMENT_DELAY": "LOW",
}


# ============================================================
# RECOMMENDED ACTIONS
# ============================================================

ACTIONS = {

    "MISSING_PAYMENT":
        "Verify payment gateway records and confirm whether the payment was received.",

    "MISSING_SETTLEMENT":
        "Verify the settlement batch and check whether the transaction is still pending.",

    "AMOUNT_MISMATCH":
        "Compare the expected and actual settlement amounts and investigate the difference.",

    "FAILED_PAYMENT":
        "Verify the payment status and determine whether a retry or customer follow-up is required.",

    "DUPLICATE_PAYMENT":
        "Verify the duplicate payment records and determine whether a reversal or refund is required.",

    "DUPLICATE_SETTLEMENT":
        "Verify the duplicate settlement entries and investigate possible over-settlement.",

    "MULTIPLE_PAYMENTS":
        "Verify why multiple payments exist for the same transaction and confirm the correct payment record.",

    "PARTIAL_SETTLEMENT":
        "Verify whether the remaining amount is pending or was incorrectly settled.",

    "REFUND":
        "Verify the refund authorization and confirm that the refund amount is correct.",

    "CHARGEBACK":
        "Review the chargeback and associated dispute record.",

    "INCORRECT_FEE":
        "Compare the charged settlement fee with the expected fee structure.",

    "UNKNOWN_ADJUSTMENT":
        "Review the adjustment entry and identify its source.",

    "WRONG_TRANSACTION_REFERENCE":
        "Verify the bank reference and correct the transaction mapping if required.",

    "DATE_MISMATCH":
        "Verify the payment processing date against the source payment record.",

    "SETTLEMENT_DELAY":
        "Check settlement processing status and investigate the settlement delay.",
}


# ============================================================
# HELPERS
# ============================================================

def money(value):

    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0

    return f"₹{value:,.2f}"


def join_values(series):

    values = []

    for value in series.dropna():

        value = str(value).strip()

        if value and value not in values:
            values.append(value)

    return "|".join(values)


# ============================================================
# LOAD EVIDENCE
# ============================================================

def load_evidence():

    # --------------------------------------------------------
    # RAW DATA
    # --------------------------------------------------------

    raw_payments = pd.read_csv(
        RAW_DIR / "payments.csv"
    )

    raw_settlements = pd.read_csv(
        RAW_DIR / "settlements.csv"
    )

    raw_bank = pd.read_csv(
        RAW_DIR / "bank_transactions.csv"
    )

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    clean_payments = pd.read_csv(
        CLEAN_DIR / "payments.csv"
    )

    clean_settlements = pd.read_csv(
        CLEAN_DIR / "settlements.csv"
    )

    # --------------------------------------------------------
    # PAYMENT EVIDENCE
    # --------------------------------------------------------

    raw_payment_group = raw_payments.groupby(
        "transaction_id",
        dropna=False
    )

    payment_evidence = raw_payment_group.agg(

        payment_count=(
            "payment_id",
            "count"
        ),

        payment_total=(
            "payment_amount",
            "sum"
        ),

        payment_statuses=(
            "payment_status",
            join_values
        ),

        payment_dates=(
            "payment_date",
            join_values
        ),

        payment_ids=(
            "payment_id",
            join_values
        ),

    ).reset_index()

    # --------------------------------------------------------
    # CLEAN PAYMENT EVIDENCE
    # --------------------------------------------------------

    clean_payment_group = clean_payments.groupby(
        "transaction_id",
        dropna=False
    )

    clean_payment_evidence = clean_payment_group.agg(

        clean_payment_total=(
            "payment_amount",
            "sum"
        ),

        clean_payment_dates=(
            "payment_date",
            join_values
        ),

    ).reset_index()

    # --------------------------------------------------------
    # BANK EVIDENCE
    # --------------------------------------------------------

    bank_group = raw_bank.groupby(
        "transaction_id",
        dropna=False
    )

    bank_evidence = bank_group.agg(

        bank_references=(
            "reference",
            join_values
        ),

        bank_total=(
            "credit_amount",
            "sum"
        ),

    ).reset_index()

    # --------------------------------------------------------
    # RAW SETTLEMENT EVIDENCE
    # --------------------------------------------------------

    raw_settlement_group = raw_settlements.groupby(
        "transaction_id",
        dropna=False
    )

    settlement_evidence = raw_settlement_group.agg(

        settlement_count=(
            "settlement_id",
            "count"
        ),

        settlement_gross_total=(
            "gross_amount",
            "sum"
        ),

        settlement_fee_total=(
            "fee",
            "sum"
        ),

        settlement_tax_total=(
            "tax",
            "sum"
        ),

        settlement_adjustment_total=(
            "adjustment",
            "sum"
        ),

        settlement_refund_total=(
            "refund",
            "sum"
        ),

        settlement_chargeback_total=(
            "chargeback",
            "sum"
        ),

        settlement_net_total=(
            "net_amount",
            "sum"
        ),

        settlement_dates=(
            "settlement_date",
            join_values
        ),

        settlement_ids=(
            "settlement_id",
            join_values
        ),

    ).reset_index()

    # --------------------------------------------------------
    # CLEAN SETTLEMENT EVIDENCE
    # --------------------------------------------------------

    clean_settlement_group = clean_settlements.groupby(
        "transaction_id",
        dropna=False
    )

    clean_settlement_evidence = clean_settlement_group.agg(

        clean_settlement_net=(
            "net_amount",
            "sum"
        ),

        clean_settlement_gross=(
            "gross_amount",
            "sum"
        ),

        clean_settlement_fee=(
            "fee",
            "sum"
        ),

        clean_settlement_tax=(
            "tax",
            "sum"
        ),

        clean_settlement_adjustment=(
            "adjustment",
            "sum"
        ),

        clean_settlement_refund=(
            "refund",
            "sum"
        ),

        clean_settlement_chargeback=(
            "chargeback",
            "sum"
        ),

        clean_settlement_date=(
            "settlement_date",
            join_values
        ),

    ).reset_index()

    # --------------------------------------------------------
    # MERGE ALL EVIDENCE
    # --------------------------------------------------------

    evidence = payment_evidence.merge(
        clean_payment_evidence,
        on="transaction_id",
        how="outer"
    )

    evidence = evidence.merge(
        bank_evidence,
        on="transaction_id",
        how="outer"
    )

    evidence = evidence.merge(
        settlement_evidence,
        on="transaction_id",
        how="outer"
    )

    evidence = evidence.merge(
        clean_settlement_evidence,
        on="transaction_id",
        how="outer"
    )

    # --------------------------------------------------------
    # NUMERIC FIELDS
    # --------------------------------------------------------

    numeric_columns = [

        "payment_count",
        "payment_total",
        "clean_payment_total",

        "bank_total",

        "settlement_count",
        "settlement_gross_total",
        "settlement_fee_total",
        "settlement_tax_total",
        "settlement_adjustment_total",
        "settlement_refund_total",
        "settlement_chargeback_total",
        "settlement_net_total",

        "clean_settlement_net",
        "clean_settlement_gross",
        "clean_settlement_fee",
        "clean_settlement_tax",
        "clean_settlement_adjustment",
        "clean_settlement_refund",
        "clean_settlement_chargeback",
    ]

    for column in numeric_columns:

        if column not in evidence.columns:
            evidence[column] = 0.0

        evidence[column] = (
            pd.to_numeric(
                evidence[column],
                errors="coerce"
            )
            .fillna(0.0)
        )

    # --------------------------------------------------------
    # STRING FIELDS
    # --------------------------------------------------------

    string_columns = [

        "payment_statuses",
        "payment_dates",
        "clean_payment_dates",

        "bank_references",

        "settlement_dates",
        "clean_settlement_date",

        "payment_ids",
        "settlement_ids",
    ]

    for column in string_columns:

        if column not in evidence.columns:
            evidence[column] = ""

        evidence[column] = (
            evidence[column]
            .fillna("")
            .astype(str)
        )

    return evidence


# ============================================================
# GENERATE EXPLANATION
# ============================================================

def generate_explanation(row):

    exception_type = row["exception_type"]

    transaction_id = row["transaction_id"]

    difference = row.get(
        "difference",
        0
    )

    # ========================================================
    # MATCHED
    # ========================================================

    if exception_type == "MATCHED":

        return {

            "severity": "NONE",

            "explanation":
                "Payment, bank and settlement records "
                "passed all reconciliation checks.",

            "evidence":
                "No reconciliation exception was detected.",

            "recommended_action":
                "No action required.",
        }

    # ========================================================
    # MISSING PAYMENT
    # ========================================================

    if exception_type == "MISSING_PAYMENT":

        explanation = (
            f"No payment record was found for "
            f"transaction {transaction_id}."
        )

        evidence = (
            f"Payment records found: "
            f"{int(row['payment_count'])}; "
            f"order/payment amount available for "
            f"reconciliation."
        )

    # ========================================================
    # MISSING SETTLEMENT
    # ========================================================

    elif exception_type == "MISSING_SETTLEMENT":

        explanation = (
            f"No settlement record was found for "
            f"transaction {transaction_id}."
        )

        evidence = (
            f"Settlement records found: "
            f"{int(row['settlement_count'])}; "
            f"payment total: "
            f"{money(row['payment_total'])}."
        )

    # ========================================================
    # AMOUNT MISMATCH
    # ========================================================

    elif exception_type == "AMOUNT_MISMATCH":

        explanation = (
            "The settlement amount does not match "
            "the clean reference amount."
        )

        evidence = (
            f"Expected settlement: "
            f"{money(row['clean_settlement_net'])}; "
            f"actual settlement: "
            f"{money(row['settlement_net_total'])}; "
            f"difference: "
            f"{money(abs(difference))}."
        )

    # ========================================================
    # FAILED PAYMENT
    # ========================================================

    elif exception_type == "FAILED_PAYMENT":

        explanation = (
            f"The payment associated with "
            f"transaction {transaction_id} "
            f"is marked as FAILED."
        )

        evidence = (
            f"Payment total: "
            f"{money(row['payment_total'])}; "
            f"payment status: "
            f"{row['payment_statuses']}."
        )

    # ========================================================
    # DUPLICATE PAYMENT
    # ========================================================

    elif exception_type == "DUPLICATE_PAYMENT":

        explanation = (
            f"Multiple payment records were found "
            f"for transaction {transaction_id}, "
            f"including a duplicate payment."
        )

        evidence = (
            f"Payment records: "
            f"{int(row['payment_count'])}; "
            f"total payment amount: "
            f"{money(row['payment_total'])}."
        )

    # ========================================================
    # DUPLICATE SETTLEMENT
    # ========================================================

    elif exception_type == "DUPLICATE_SETTLEMENT":

        explanation = (
            f"Multiple settlement records were found "
            f"for transaction {transaction_id}, "
            f"including a duplicate settlement."
        )

        evidence = (
            f"Settlement records: "
            f"{int(row['settlement_count'])}; "
            f"total settlement amount: "
            f"{money(row['settlement_net_total'])}."
        )

    # ========================================================
    # MULTIPLE PAYMENTS
    # ========================================================

    elif exception_type == "MULTIPLE_PAYMENTS":

        explanation = (
            f"Multiple payment records exist for "
            f"transaction {transaction_id}."
        )

        evidence = (
            f"Payment records: "
            f"{int(row['payment_count'])}; "
            f"total payment amount: "
            f"{money(row['payment_total'])}."
        )

    # ========================================================
    # PARTIAL SETTLEMENT
    # ========================================================

    elif exception_type == "PARTIAL_SETTLEMENT":

        explanation = (
            "The settlement amount is lower than "
            "the expected gross settlement amount."
        )

        evidence = (
            f"Gross settlement: "
            f"{money(row['settlement_gross_total'])}; "
            f"actual net settlement: "
            f"{money(row['settlement_net_total'])}; "
            f"difference: "
            f"{money(abs(difference))}."
        )

    # ========================================================
    # REFUND
    # ========================================================

    elif exception_type == "REFUND":

        explanation = (
            f"A refund was recorded for "
            f"transaction {transaction_id}."
        )

        evidence = (
            f"Refund amount: "
            f"{money(row['settlement_refund_total'])}."
        )

    # ========================================================
    # CHARGEBACK
    # ========================================================

    elif exception_type == "CHARGEBACK":

        explanation = (
            f"A chargeback was recorded for "
            f"transaction {transaction_id}."
        )

        evidence = (
            f"Chargeback amount: "
            f"{money(row['settlement_chargeback_total'])}."
        )

    # ========================================================
    # INCORRECT FEE
    # ========================================================

    elif exception_type == "INCORRECT_FEE":

        explanation = (
            "The settlement fee differs from "
            "the expected fee structure."
        )

        evidence = (
            f"Gross amount: "
            f"{money(row['settlement_gross_total'])}; "
            f"fee charged: "
            f"{money(row['settlement_fee_total'])}."
        )

    # ========================================================
    # UNKNOWN ADJUSTMENT
    # ========================================================

    elif exception_type == "UNKNOWN_ADJUSTMENT":

        explanation = (
            f"An unexpected settlement adjustment "
            f"was recorded for transaction "
            f"{transaction_id}."
        )

        evidence = (
            f"Adjustment amount: "
            f"{money(row['settlement_adjustment_total'])}."
        )

    # ========================================================
    # WRONG TRANSACTION REFERENCE
    # ========================================================

    elif exception_type == "WRONG_TRANSACTION_REFERENCE":

        explanation = (
            f"The bank reference does not match "
            f"transaction {transaction_id}."
        )

        evidence = (
            f"Bank reference: "
            f"{row['bank_references']}."
        )

    # ========================================================
    # DATE MISMATCH
    # ========================================================

    elif exception_type == "DATE_MISMATCH":

        explanation = (
            "The payment date differs from "
            "the clean reference payment date."
        )

        evidence = (
            f"Actual payment date: "
            f"{row['payment_dates']}; "
            f"reference payment date: "
            f"{row['clean_payment_dates']}."
        )

    # ========================================================
    # SETTLEMENT DELAY
    # ========================================================

    elif exception_type == "SETTLEMENT_DELAY":

        explanation = (
            "The settlement date differs from "
            "the clean reference settlement date."
        )

        evidence = (
            f"Actual settlement date: "
            f"{row['settlement_dates']}; "
            f"reference settlement date: "
            f"{row['clean_settlement_date']}."
        )

    # ========================================================
    # UNKNOWN
    # ========================================================

    else:

        explanation = (
            f"An unknown reconciliation exception "
            f"was detected: {exception_type}."
        )

        evidence = (
            "Review the transaction records manually."
        )

    return {

        "severity":
            SEVERITY.get(
                exception_type,
                "MEDIUM"
            ),

        "explanation":
            explanation,

        "evidence":
            evidence,

        "recommended_action":
            ACTIONS.get(
                exception_type,
                "Review the transaction manually."
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    input_file = (
        RECONCILIATION_DIR
        / "reconciliation_results.csv"
    )

    output_file = (
        RECONCILIATION_DIR
        / "exception_explanations.csv"
    )

    # --------------------------------------------------------
    # LOAD RECONCILIATION RESULTS
    # --------------------------------------------------------

    if not input_file.exists():

        raise FileNotFoundError(
            f"Reconciliation results not found:\n"
            f"{input_file}"
        )

    results = pd.read_csv(
        input_file
    )

    print(
        "========== LOADING PHASE 4 DATA =========="
    )

    print(
        f"Reconciliation results: "
        f"{len(results)} rows"
    )

    # --------------------------------------------------------
    # LOAD RAW/CLEAN EVIDENCE
    # --------------------------------------------------------

    evidence = load_evidence()

    print(
        f"Evidence records: "
        f"{len(evidence)}"
    )

    # --------------------------------------------------------
    # MERGE RESULTS + EVIDENCE
    # --------------------------------------------------------

    # --------------------------------------------------------
    # MERGE ONLY NEW EVIDENCE COLUMNS
    # --------------------------------------------------------
    # reconciliation_results.csv already contains many fields
    # such as payment_count, settlement_count, payment_total,
    # settlement_net_total, etc.
    #
    # Therefore, only add evidence columns that are not already
    # present in the reconciliation results.

    new_evidence_columns = [
        column
        for column in evidence.columns
        if column == "transaction_id"
        or column not in results.columns
    ]

    evidence_to_merge = evidence[
        new_evidence_columns
    ].copy()

    results = results.merge(
        evidence_to_merge,
        on="transaction_id",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if len(results) != 1000:

        raise ValueError(
            "Phase 4 should contain "
            "1000 transaction records."
        )

    if results["transaction_id"].nunique() != 1000:

        raise ValueError(
            "Duplicate transaction IDs "
            "found after Phase 4 merge."
        )

    print(
        "Phase 4 evidence merge successful."
    )

    # --------------------------------------------------------
    # GENERATE EXPLANATIONS
    # --------------------------------------------------------

    explanations = []

    for _, row in results.iterrows():

        explanation = generate_explanation(
            row
        )

        explanations.append({

            "transaction_id":
                row["transaction_id"],

            "status":
                row["status"],

            "exception_type":
                row["exception_type"],

            "severity":
                explanation["severity"],

            "explanation":
                explanation["explanation"],

            "evidence":
                explanation["evidence"],

            "difference":
                row["difference"],

            "recommended_action":
                explanation["recommended_action"],
        })

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = pd.DataFrame(
        explanations
    )

    output.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\n========== PHASE 4 VALIDATION =========="
    )

    print(
        f"Input records: "
        f"{len(results)}"
    )

    print(
        f"Explanation records: "
        f"{len(output)}"
    )

    print(
        "\nSeverity distribution:"
    )

    print(
        output["severity"]
        .value_counts()
    )

    print(
        "\nException distribution:"
    )

    print(
        output["exception_type"]
        .value_counts()
    )

    print(
        f"\nExplanation file saved to:\n"
        f"{output_file}"
    )

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    print(
        "\n========== SAMPLE EXPLANATIONS =========="
    )

    print(
        output[
            [
                "transaction_id",
                "exception_type",
                "severity",
                "explanation",
                "recommended_action",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\n================================================"
    )

    print(
        "       PHASE 4 COMPLETED"
    )

    print(
        "================================================"
    )


if __name__ == "__main__":
    main()