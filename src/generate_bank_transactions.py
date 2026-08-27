from pathlib import Path
from datetime import datetime, timedelta
import random

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent

MASTER_FILE = BASE_DIR / "data" / "master_transactions.csv"
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "bank_transactions.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MASTER TRANSACTIONS
# ============================================================

def load_master_transactions():

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master dataset not found at:\n{MASTER_FILE}\n\n"
            "Run generate_data.py first."
        )

    df = pd.read_csv(MASTER_FILE)

    print(f"Loaded master transactions: {len(df)} rows")

    return df


# ============================================================
# GENERATE BANK TRANSACTION DATE
# ============================================================

def generate_bank_transaction_date(order_date):

    """
    Generate the date on which the bank records the
    transaction.

    For the clean baseline dataset, the bank transaction
    occurs on the same day or shortly after the order.
    """

    order_date = datetime.strptime(
        order_date,
        "%Y-%m-%d"
    )

    delay_days = random.choice([
        0,
        0,
        0,
        1,
        1,
        2
    ])

    bank_date = order_date + timedelta(
        days=delay_days
    )

    return bank_date.strftime("%Y-%m-%d")


# ============================================================
# GENERATE BANK TRANSACTIONS
# ============================================================

def generate_bank_transactions(master_df):

    bank_transactions = []

    for _, transaction in master_df.iterrows():

        transaction_id = transaction["transaction_id"]

        bank_transaction_id = (
            f"BANK{transaction_id[3:]}"
        )

        bank_date = generate_bank_transaction_date(
            transaction["order_date"]
        )

        bank_transactions.append({
            "bank_transaction_id": bank_transaction_id,
            "transaction_id": transaction_id,
            "merchant_id": transaction["merchant_id"],
            "transaction_date": bank_date,
            "credit_amount": transaction["order_amount"],
            "transaction_type": "CREDIT",
            "reference": transaction_id,
            "currency": transaction["currency"],
        })

    return pd.DataFrame(bank_transactions)


# ============================================================
# VALIDATE BANK TRANSACTIONS
# ============================================================

def validate_bank_transactions(
    bank_df,
    master_df
):

    print("\n========== BANK TRANSACTIONS VALIDATION ==========")

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    print(
        f"Total bank transactions: {len(bank_df)}"
    )

    assert len(bank_df) == len(master_df)

    # --------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------

    print(
        f"Unique bank transaction IDs: "
        f"{bank_df['bank_transaction_id'].nunique()}"
    )

    print(
        f"Unique transaction IDs: "
        f"{bank_df['transaction_id'].nunique()}"
    )

    assert bank_df[
        "bank_transaction_id"
    ].is_unique

    assert bank_df[
        "transaction_id"
    ].is_unique

    # --------------------------------------------------------
    # Transaction type
    # --------------------------------------------------------

    print("\nTransaction types:")

    print(
        bank_df[
            "transaction_type"
        ].value_counts()
    )

    assert (
        bank_df["transaction_type"] == "CREDIT"
    ).all()

    # --------------------------------------------------------
    # Amount validation
    # --------------------------------------------------------

    assert (
        bank_df["credit_amount"] > 0
    ).all()

    print(
        f"\nTotal bank credit: "
        f"₹{bank_df['credit_amount'].sum():,.2f}"
    )

    # --------------------------------------------------------
    # Currency
    # --------------------------------------------------------

    assert (
        bank_df["currency"] == "INR"
    ).all()

    # --------------------------------------------------------
    # Reference validation
    # --------------------------------------------------------

    assert (
        bank_df["reference"]
        == bank_df["transaction_id"]
    ).all()

    # --------------------------------------------------------
    # Compare against master dataset
    # --------------------------------------------------------

    merged = bank_df.merge(
        master_df[
            [
                "transaction_id",
                "merchant_id",
                "order_amount",
                "currency",
            ]
        ],
        on="transaction_id",
        how="left",
        suffixes=(
            "_bank",
            "_master"
        ),
        validate="one_to_one",
    )

    # Every transaction must exist in master
    assert (
        merged["order_amount"].notna()
    ).all()

    # Bank credit should equal order amount
    assert (
        merged["credit_amount"]
        == merged["order_amount"]
    ).all()

    # Merchant should match
    assert (
        merged["merchant_id_bank"]
        == merged["merchant_id_master"]
    ).all()

    # Currency should match
    assert (
        merged["currency_bank"]
        == merged["currency_master"]
    ).all()

    print(
        "\nAll bank transactions correctly "
        "match the master transactions."
    )

    print(
        "Bank transactions validation successful."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Generating bank_transactions.csv..."
    )

    master_df = load_master_transactions()

    bank_df = generate_bank_transactions(
        master_df
    )

    validate_bank_transactions(
        bank_df,
        master_df
    )

    bank_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nBank transactions dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nFirst 10 bank transactions:"
    )

    print(
        bank_df.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()