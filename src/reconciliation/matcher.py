from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
RECONCILIATION_DIR = DATA_DIR / "reconciliation"

RECONCILIATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMPORT LOADER
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from loader import load_all_datasets


# ============================================================
# HELPERS
# ============================================================

def join_unique_values(series):

    values = []

    for value in series.dropna().astype(str):

        if value not in values:
            values.append(value)

    return "|".join(values)


# ============================================================
# LOAD CLEAN REFERENCE DATA
# ============================================================

def load_clean_data():

    clean_orders = pd.read_csv(
        CLEAN_DIR / "orders.csv"
    )

    clean_payments = pd.read_csv(
        CLEAN_DIR / "payments.csv"
    )

    clean_bank = pd.read_csv(
        CLEAN_DIR / "bank_transactions.csv"
    )

    clean_settlements = pd.read_csv(
        CLEAN_DIR / "settlements.csv"
    )

    return (
        clean_orders,
        clean_payments,
        clean_bank,
        clean_settlements
    )


# ============================================================
# BUILD MATCH TABLE
# ============================================================

def build_match_table(
    orders,
    payments,
    bank,
    settlements,
    clean_orders,
    clean_payments,
    clean_bank,
    clean_settlements
):

    # ========================================================
    # BASE ORDER DATA
    # ========================================================

    match = orders[
        [
            "transaction_id",
            "order_id",
            "merchant_id",
            "customer_id",
            "order_date",
            "order_amount",
            "currency",
            "payment_method",
        ]
    ].copy()

    # ========================================================
    # RAW PAYMENTS
    # ========================================================

    payment_group = payments.groupby(
        "transaction_id",
        dropna=False
    )

    payment_summary = payment_group.agg(
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
            join_unique_values
        ),

        payment_dates=(
            "payment_date",
            join_unique_values
        ),

        payment_ids=(
            "payment_id",
            join_unique_values
        ),

        payment_methods=(
            "payment_method",
            join_unique_values
        ),
    ).reset_index()

    match = match.merge(
        payment_summary,
        on="transaction_id",
        how="left"
    )

    # ========================================================
    # RAW BANK
    # ========================================================

    bank_group = bank.groupby(
        "transaction_id",
        dropna=False
    )

    bank_summary = bank_group.agg(
        bank_transaction_count=(
            "bank_transaction_id",
            "count"
        ),

        bank_total=(
            "credit_amount",
            "sum"
        ),

        bank_references=(
            "reference",
            join_unique_values
        ),

        bank_dates=(
            "transaction_date",
            join_unique_values
        ),

        bank_transaction_types=(
            "transaction_type",
            join_unique_values
        ),
    ).reset_index()

    match = match.merge(
        bank_summary,
        on="transaction_id",
        how="left"
    )

    # ========================================================
    # RAW SETTLEMENTS
    # ========================================================

    settlement_group = settlements.groupby(
        "transaction_id",
        dropna=False
    )

    settlement_summary = settlement_group.agg(
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
            join_unique_values
        ),

        settlement_ids=(
            "settlement_id",
            join_unique_values
        ),

        settlement_references=(
            "settlement_reference",
            join_unique_values
        ),

        settlement_statuses=(
            "settlement_status",
            join_unique_values
        ),
    ).reset_index()

    match = match.merge(
        settlement_summary,
        on="transaction_id",
        how="left"
    )

    # ========================================================
    # CLEAN PAYMENT REFERENCE
    # ========================================================

    clean_payment_group = (
        clean_payments.groupby(
            "transaction_id",
            dropna=False
        )
    )

    clean_payment_summary = (
        clean_payment_group.agg(
            clean_payment_dates=(
                "payment_date",
                join_unique_values
            ),

            clean_payment_total=(
                "payment_amount",
                "sum"
            ),
        )
        .reset_index()
    )

    match = match.merge(
        clean_payment_summary,
        on="transaction_id",
        how="left"
    )

    # ========================================================
    # CLEAN SETTLEMENT REFERENCE
    # ========================================================

    clean_settlement_group = (
        clean_settlements.groupby(
            "transaction_id",
            dropna=False
        )
    )

    clean_settlement_summary = (
        clean_settlement_group.agg(
            clean_settlement_date=(
                "settlement_date",
                join_unique_values
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

            clean_settlement_net=(
                "net_amount",
                "sum"
            ),
        )
        .reset_index()
    )

    match = match.merge(
        clean_settlement_summary,
        on="transaction_id",
        how="left"
    )

    # ========================================================
    # CLEAN BANK REFERENCE
    # ========================================================

    clean_bank_group = (
        clean_bank.groupby(
            "transaction_id",
            dropna=False
        )
    )

    clean_bank_summary = (
        clean_bank_group.agg(
            clean_bank_reference=(
                "reference",
                join_unique_values
            )
        )
        .reset_index()
    )

    match = match.merge(
        clean_bank_summary,
        on="transaction_id",
        how="left"
    )

    # ========================================================
    # FILL NUMERIC VALUES
    # ========================================================

    numeric_columns = [

        "payment_count",
        "payment_total",

        "bank_transaction_count",
        "bank_total",

        "settlement_count",
        "settlement_gross_total",
        "settlement_fee_total",
        "settlement_tax_total",
        "settlement_adjustment_total",
        "settlement_refund_total",
        "settlement_chargeback_total",
        "settlement_net_total",

        "clean_payment_total",

        "clean_settlement_gross",
        "clean_settlement_fee",
        "clean_settlement_tax",
        "clean_settlement_adjustment",
        "clean_settlement_refund",
        "clean_settlement_chargeback",
        "clean_settlement_net",
    ]

    for column in numeric_columns:

        if column not in match.columns:
            match[column] = 0.0

        match[column] = (
            pd.to_numeric(
                match[column],
                errors="coerce"
            )
            .fillna(0.0)
        )

    # ========================================================
    # FILL STRING VALUES
    # ========================================================

    string_columns = [

        "payment_statuses",
        "payment_dates",
        "payment_ids",
        "payment_methods",

        "bank_references",
        "bank_dates",
        "bank_transaction_types",

        "settlement_dates",
        "settlement_ids",
        "settlement_references",
        "settlement_statuses",

        "clean_payment_dates",
        "clean_settlement_date",
        "clean_bank_reference",
    ]

    for column in string_columns:

        if column not in match.columns:
            match[column] = ""

        match[column] = (
            match[column]
            .fillna("")
            .astype(str)
        )

    # ========================================================
    # SORT
    # ========================================================

    match = (
        match
        .sort_values("transaction_id")
        .reset_index(drop=True)
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        "\n========== MATCH TABLE VALIDATION =========="
    )

    print(
        f"Match table rows: {len(match)}"
    )

    print(
        "Unique transactions:",
        match["transaction_id"].nunique()
    )

    print("\nPayment count distribution:")

    print(
        match["payment_count"]
        .value_counts()
        .sort_index()
    )

    print("\nSettlement count distribution:")

    print(
        match["settlement_count"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTransactions with no payment:",
        (match["payment_count"] == 0).sum()
    )

    print(
        "Transactions with no settlement:",
        (match["settlement_count"] == 0).sum()
    )

    print(
        "Transactions with multiple payments:",
        (match["payment_count"] > 1).sum()
    )

    print(
        "Transactions with multiple settlements:",
        (match["settlement_count"] > 1).sum()
    )

    if len(match) != 1000:
        raise ValueError(
            "Match table should contain "
            "1000 transactions."
        )

    if (
        match["transaction_id"].nunique()
        != len(match)
    ):
        raise ValueError(
            "Duplicate transaction IDs."
        )

    print(
        "\nMatch table validation successful."
    )

    return match


# ============================================================
# SAVE
# ============================================================

def save_match_table(match):

    output_file = (
        RECONCILIATION_DIR
        / "match_table.csv"
    )

    match.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nMatch table saved to:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    datasets = load_all_datasets()

    orders = datasets["orders"]
    payments = datasets["payments"]
    bank = datasets["bank_transactions"]
    settlements = datasets["settlements"]

    (
        clean_orders,
        clean_payments,
        clean_bank,
        clean_settlements
    ) = load_clean_data()

    print("\n========== CLEAN REFERENCE DATA ==========")

    print(
        f"Clean orders: {len(clean_orders)}"
    )

    print(
        f"Clean payments: {len(clean_payments)}"
    )

    print(
        f"Clean bank transactions: "
        f"{len(clean_bank)}"
    )

    print(
        f"Clean settlements: "
        f"{len(clean_settlements)}"
    )

    match = build_match_table(
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
        "\n========== MATCH TABLE SAMPLE =========="
    )

    sample_columns = [

        "transaction_id",

        "order_amount",

        "payment_total",
        "clean_payment_total",

        "settlement_net_total",
        "clean_settlement_net",

        "settlement_dates",
        "clean_settlement_date",
    ]

    print(
        match[
            sample_columns
        ].head(10).to_string(
            index=False
        )
    )

    save_match_table(match)

    return match


if __name__ == "__main__":
    main()