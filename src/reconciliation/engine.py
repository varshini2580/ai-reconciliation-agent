from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RECONCILIATION_DIR = (
    DATA_DIR / "reconciliation"
)

RECONCILIATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMPORTS
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from loader import load_all_datasets
from matcher import build_match_table, load_clean_data
from rules import apply_rules


# ============================================================
# APPLY RULES
# ============================================================

def reconcile(match_table):

    results = []

    for _, row in match_table.iterrows():

        result = apply_rules(row)

        result_row = {
            "transaction_id":
                row["transaction_id"],

            "order_id":
                row["order_id"],

            "order_amount":
                row["order_amount"],

            "payment_count":
                row["payment_count"],

            "payment_total":
                row["payment_total"],

            "payment_dates":
                row["payment_dates"],

            "clean_payment_dates":
                row["clean_payment_dates"],

            "bank_references":
                row["bank_references"],

            "settlement_dates":
                row["settlement_dates"],

            "clean_settlement_date":
                row["clean_settlement_date"],

            "bank_transaction_count":
                row["bank_transaction_count"],

            "bank_total":
                row["bank_total"],

            "settlement_count":
                row["settlement_count"],

            "settlement_gross_total":
                row["settlement_gross_total"],

            "settlement_fee_total":
                row["settlement_fee_total"],

            "settlement_tax_total":
                row["settlement_tax_total"],

            "settlement_adjustment_total":
                row[
                    "settlement_adjustment_total"
                ],

            "settlement_refund_total":
                row[
                    "settlement_refund_total"
                ],

            "settlement_chargeback_total":
                row[
                    "settlement_chargeback_total"
                ],

            "settlement_net_total":
                row[
                    "settlement_net_total"
                ],

            "status":
                (
                    "MATCHED"
                    if result["exception_type"]
                    == "MATCHED"
                    else "EXCEPTION"
                ),

            "exception_type":
                result["exception_type"],

            "reason":
                result["reason"],

            "difference":
                result["difference"],

            "clean_settlement_net":
                row["clean_settlement_net"],

            "clean_settlement_gross":
                row["clean_settlement_gross"],

            "clean_settlement_fee":
                row["clean_settlement_fee"],

            "clean_settlement_tax":
                row["clean_settlement_tax"],

            "clean_settlement_adjustment":
                row["clean_settlement_adjustment"],

            "clean_settlement_refund":
                row["clean_settlement_refund"],

            "clean_settlement_chargeback":
                row["clean_settlement_chargeback"],
        }

        results.append(result_row)

    return pd.DataFrame(results)


# ============================================================
# VALIDATE RESULTS
# ============================================================

def validate_results(results):

    print(
        "\n========== RECONCILIATION VALIDATION =========="
    )

    print(
        "Result rows:",
        len(results)
    )

    print(
        "Unique transactions:",
        results["transaction_id"].nunique()
    )

    if len(results) != 1000:
        raise ValueError(
            "Expected 1000 reconciliation results."
        )

    if (
        results["transaction_id"].nunique()
        != len(results)
    ):
        raise ValueError(
            "Duplicate transaction IDs found."
        )

    print("\nStatus distribution:")

    print(
        results["status"]
        .value_counts()
    )

    print("\nDetected exception types:")

    print(
        results["exception_type"]
        .value_counts()
    )

    if results["status"].isna().any():

        raise ValueError(
            "Some transactions were not classified."
        )

    print(
        "\nAll transactions classified."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    output_file = (
        RECONCILIATION_DIR
        / "reconciliation_results.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nReconciliation results saved to:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n================================================"
    )

    print(
        "        RECONCILIATION ENGINE"
    )

    print(
        "================================================"
    )

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    print(
        "\n[1/3] Loading raw datasets..."
    )

    datasets = load_all_datasets()

    orders = datasets["orders"]
    payments = datasets["payments"]
    bank = datasets["bank_transactions"]
    settlements = datasets["settlements"]

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------
    clean_orders, clean_payments, clean_bank, clean_settlements = load_clean_data()

    match_table = build_match_table(
        orders,
        payments,
        bank,
        settlements,
        clean_orders,
        clean_payments,
        clean_bank,
        clean_settlements
    )

    print(
        "\n[2/3] Building transaction match table..."
    )

    print(
        f"Match table created: "
        f"{len(match_table)} transactions"
    )

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    print(
        "\n[3/3] Applying reconciliation rules..."
    )

    results = reconcile(
        match_table
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_results(
        results
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print(
        "\n========== SAMPLE RESULTS =========="
    )

    sample_columns = [
        "transaction_id",
        "order_amount",
        "payment_total",
        "bank_total",
        "settlement_net_total",
        "status",
        "exception_type",
        "difference",
    ]

    print(
        results[
            sample_columns
        ].head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results
    )

    print(
        "\n================================================"
    )

    print(
        "     RECONCILIATION ENGINE COMPLETED"
    )

    print(
        "================================================"
    )


if __name__ == "__main__":
    main()