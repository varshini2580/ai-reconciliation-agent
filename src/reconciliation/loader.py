from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"


# ============================================================
# REQUIRED SCHEMAS
# ============================================================

REQUIRED_COLUMNS = {

    "orders": [
        "order_id",
        "transaction_id",
        "merchant_id",
        "customer_id",
        "order_date",
        "order_amount",
        "currency",
        "payment_method",
    ],

    "payments": [
        "payment_id",
        "transaction_id",
        "order_id",
        "merchant_id",
        "payment_date",
        "payment_amount",
        "payment_status",
        "payment_method",
        "currency",
    ],

    "bank_transactions": [
        "bank_transaction_id",
        "transaction_id",
        "merchant_id",
        "transaction_date",
        "credit_amount",
        "transaction_type",
        "reference",
        "currency",
    ],

    "settlements": [
        "settlement_id",
        "transaction_id",
        "merchant_id",
        "settlement_date",
        "gross_amount",
        "fee",
        "tax",
        "adjustment",
        "refund",
        "chargeback",
        "net_amount",
        "settlement_status",
        "settlement_reference",
        "currency",
    ],
}


# ============================================================
# LOAD SINGLE DATASET
# ============================================================

def load_dataset(
    filename,
    dataset_name
):

    file_path = RAW_DIR / filename

    if not file_path.exists():

        raise FileNotFoundError(
            f"{dataset_name} dataset not found:\n"
            f"{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    print(
        f"Loaded {dataset_name}: "
        f"{len(df)} rows"
    )

    return df


# ============================================================
# VALIDATE SCHEMA
# ============================================================

def validate_schema(
    df,
    dataset_name
):

    required_columns = (
        REQUIRED_COLUMNS[
            dataset_name
        ]
    )

    actual_columns = set(
        df.columns
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in actual_columns
    ]

    if missing_columns:

        raise ValueError(
            f"{dataset_name} is missing "
            f"required columns:\n"
            f"{missing_columns}"
        )

    print(
        f"[OK] {dataset_name} schema valid"
    )


# ============================================================
# VALIDATE TRANSACTION IDs
# ============================================================

def validate_transaction_ids(
    df,
    dataset_name
):

    if "transaction_id" not in df.columns:

        raise ValueError(
            f"{dataset_name} does not contain "
            f"transaction_id"
        )

    missing_ids = (
        df["transaction_id"]
        .isna()
        .sum()
    )

    if missing_ids > 0:

        raise ValueError(
            f"{dataset_name} contains "
            f"{missing_ids} missing transaction IDs"
        )

    print(
        f"[OK] {dataset_name} transaction IDs valid"
    )


# ============================================================
# LOAD ALL DATASETS
# ============================================================

def load_all_datasets():

    print(
        "\n========== LOADING RAW DATA =========="
    )

    orders = load_dataset(
        "orders.csv",
        "orders"
    )

    payments = load_dataset(
        "payments.csv",
        "payments"
    )

    bank_transactions = load_dataset(
        "bank_transactions.csv",
        "bank_transactions"
    )

    settlements = load_dataset(
        "settlements.csv",
        "settlements"
    )

    datasets = {

        "orders":
            orders,

        "payments":
            payments,

        "bank_transactions":
            bank_transactions,

        "settlements":
            settlements,
    }

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    print(
        "\n========== SCHEMA VALIDATION =========="
    )

    for dataset_name, df in datasets.items():

        validate_schema(
            df,
            dataset_name
        )

    # --------------------------------------------------------
    # Transaction ID validation
    # --------------------------------------------------------

    print(
        "\n========== TRANSACTION ID VALIDATION =========="
    )

    for dataset_name, df in datasets.items():

        validate_transaction_ids(
            df,
            dataset_name
        )

    return datasets


# ============================================================
# MAIN - TEST LOADER
# ============================================================

def main():

    datasets = load_all_datasets()

    print(
        "\n========== DATASET SUMMARY =========="
    )

    for name, df in datasets.items():

        print(
            f"{name}: "
            f"{len(df)} rows, "
            f"{len(df.columns)} columns"
        )

    print(
        "\nLoader validation successful."
    )


if __name__ == "__main__":
    main()