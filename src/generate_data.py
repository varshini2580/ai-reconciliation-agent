import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
NUM_TRANSACTIONS = 1000

random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

RAW_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MASTER TRANSACTION GENERATOR
# ============================================================

def generate_master_transactions(num_transactions=NUM_TRANSACTIONS):
    """
    Generate the master set of business transactions.

    These transactions represent the true underlying financial
    activity. The individual CSV files will later be generated
    from these transactions.
    """

    transactions = []

    start_date = datetime(2026, 1, 1)

    merchants = [
        "MERCHANT_001",
        "MERCHANT_002",
        "MERCHANT_003",
        "MERCHANT_004",
        "MERCHANT_005",
    ]

    payment_methods = [
        "UPI",
        "CARD",
        "NETBANKING",
        "WALLET",
    ]

    for i in range(1, num_transactions + 1):

        transaction_id = f"TXN{i:05d}"
        order_id = f"ORD{i:05d}"
        customer_id = f"CUST{random.randint(1, 300):04d}"
        merchant_id = random.choice(merchants)

        # Generate transaction date within the first 6 months
        # of 2026.
        order_date = start_date + timedelta(
            days=random.randint(0, 180)
        )

        # Generate realistic order amounts.
        order_amount = random.choice([
            199,
            299,
            499,
            799,
            999,
            1499,
            1999,
            2499,
            2999,
            3999,
            4999,
            5999,
            7999,
            9999,
        ])

        payment_method = random.choice(payment_methods)

        transactions.append({
            "transaction_id": transaction_id,
            "order_id": order_id,
            "customer_id": customer_id,
            "merchant_id": merchant_id,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "order_amount": order_amount,
            "currency": "INR",
            "payment_method": payment_method,
        })

    return pd.DataFrame(transactions)


# ============================================================
# VALIDATION
# ============================================================

def validate_master_transactions(df):
    """
    Validate the master transaction dataset.
    """

    print("\n========== MASTER DATA VALIDATION ==========")

    print(f"Total transactions: {len(df)}")

    print(
        f"Unique transaction IDs: "
        f"{df['transaction_id'].nunique()}"
    )

    print(
        f"Unique order IDs: "
        f"{df['order_id'].nunique()}"
    )

    print(
        f"Total transaction value: "
        f"₹{df['order_amount'].sum():,.2f}"
    )

    print("\nPayment methods:")
    print(df["payment_method"].value_counts())

    print("\nMerchants:")
    print(df["merchant_id"].value_counts())

    # Basic assertions
    assert len(df) == NUM_TRANSACTIONS
    assert df["transaction_id"].is_unique
    assert df["order_id"].is_unique
    assert (df["order_amount"] > 0).all()
    assert (df["currency"] == "INR").all()

    print("\nValidation successful.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("Generating master transactions...")

    master_df = generate_master_transactions()

    validate_master_transactions(master_df)

    # Save master dataset temporarily.
    # This is our internal source for generating the
    # four financial datasets.
    output_path = DATA_DIR / "master_transactions.csv"

    master_df.to_csv(output_path, index=False)

    print(
        f"\nMaster dataset saved to:\n"
        f"{output_path}"
    )

    print("\nFirst 10 transactions:")
    print(master_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()