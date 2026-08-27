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
OUTPUT_FILE = OUTPUT_DIR / "payments.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MASTER DATA
# ============================================================

def load_master_transactions():
    """
    Load the master transaction dataset.
    """

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master dataset not found at:\n{MASTER_FILE}\n\n"
            "Run generate_data.py first."
        )

    df = pd.read_csv(MASTER_FILE)

    print(f"Loaded master transactions: {len(df)} rows")

    return df


# ============================================================
# GENERATE PAYMENT DATE
# ============================================================

def generate_payment_date(order_date):
    """
    Generate a payment date.

    For the clean baseline dataset, payment happens either
    on the same day or shortly after the order.
    """

    order_date = datetime.strptime(
        order_date,
        "%Y-%m-%d"
    )

    delay_days = random.choice([0, 0, 0, 1, 1, 2])

    payment_date = order_date + timedelta(
        days=delay_days
    )

    return payment_date.strftime("%Y-%m-%d")


# ============================================================
# GENERATE PAYMENTS
# ============================================================

def generate_payments(master_df):
    """
    Generate one clean payment record for every master
    transaction.

    No exceptions are injected at this stage.
    """

    payments = []

    for _, transaction in master_df.iterrows():

        payment_id = (
            f"PAY{transaction['transaction_id'][3:]}"
        )

        payment_date = generate_payment_date(
            transaction["order_date"]
        )

        payments.append({
            "payment_id": payment_id,
            "transaction_id": transaction["transaction_id"],
            "order_id": transaction["order_id"],
            "merchant_id": transaction["merchant_id"],
            "payment_date": payment_date,
            "payment_amount": transaction["order_amount"],
            "payment_status": "SUCCESS",
            "payment_method": transaction["payment_method"],
            "currency": transaction["currency"],
        })

    return pd.DataFrame(payments)


# ============================================================
# VALIDATE PAYMENTS
# ============================================================

def validate_payments(payments_df, master_df):
    """
    Validate the generated payment dataset.
    """

    print("\n========== PAYMENTS VALIDATION ==========")

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    print(f"Total payments: {len(payments_df)}")

    assert len(payments_df) == len(master_df)

    # --------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------

    print(
        f"Unique payment IDs: "
        f"{payments_df['payment_id'].nunique()}"
    )

    print(
        f"Unique transaction IDs: "
        f"{payments_df['transaction_id'].nunique()}"
    )

    assert payments_df["payment_id"].is_unique
    assert payments_df["transaction_id"].is_unique

    # --------------------------------------------------------
    # Payment status
    # --------------------------------------------------------

    print("\nPayment status:")
    print(
        payments_df["payment_status"].value_counts()
    )

    assert (
        payments_df["payment_status"] == "SUCCESS"
    ).all()

    # --------------------------------------------------------
    # Payment amounts
    # --------------------------------------------------------

    assert (
        payments_df["payment_amount"] > 0
    ).all()

    print(
        f"\nTotal payment value: "
        f"₹{payments_df['payment_amount'].sum():,.2f}"
    )

    # --------------------------------------------------------
    # Currency
    # --------------------------------------------------------

    assert (
        payments_df["currency"] == "INR"
    ).all()

    # --------------------------------------------------------
    # Verify payment amounts against master
    # --------------------------------------------------------

    merged = payments_df.merge(
        master_df[
            [
                "transaction_id",
                "order_id",
                "merchant_id",
                "order_amount",
                "payment_method",
                "currency",
            ]
        ],
        on="transaction_id",
        how="left",
        suffixes=("_payment", "_master"),
        validate="one_to_one",
    )

    # No transaction should be missing
    assert merged["order_amount"].notna().all()

    # Payment should equal order amount
    assert (
        merged["payment_amount"]
        == merged["order_amount"]
    ).all()

    # Related IDs should match
    assert (
        merged["order_id_payment"]
        == merged["order_id_master"]
    ).all()

    assert (
        merged["merchant_id_payment"]
        == merged["merchant_id_master"]
    ).all()

    assert (
        merged["payment_method_payment"]
        == merged["payment_method_master"]
    ).all()

    assert (
        merged["currency_payment"]
        == merged["currency_master"]
    ).all()

    print(
        "\nAll payments correctly match "
        "the master transactions."
    )

    print("Payments validation successful.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("Generating payments.csv...")

    # Load master data
    master_df = load_master_transactions()

    # Generate payments
    payments_df = generate_payments(
        master_df
    )

    # Validate
    validate_payments(
        payments_df,
        master_df
    )

    # Save
    payments_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nPayments dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # Preview
    print("\nFirst 10 payments:")

    print(
        payments_df.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()