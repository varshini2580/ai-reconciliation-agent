from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MASTER_FILE = BASE_DIR / "data" / "master_transactions.csv"
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "orders.csv"


# Make sure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MASTER TRANSACTIONS
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
# GENERATE ORDERS
# ============================================================

def generate_orders(master_df):
    """
    Create the orders dataset from the master transactions.

    At this stage we create clean, correct order records.
    Exceptions will be introduced later in the pipeline.
    """

    orders_df = master_df[
        [
            "order_id",
            "transaction_id",
            "merchant_id",
            "customer_id",
            "order_date",
            "order_amount",
            "currency",
            "payment_method",
        ]
    ].copy()

    return orders_df


# ============================================================
# VALIDATE ORDERS
# ============================================================

def validate_orders(orders_df, master_df):
    """
    Validate that the generated orders correctly represent
    the master transactions.
    """

    print("\n========== ORDERS VALIDATION ==========")

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    print(f"Total orders: {len(orders_df)}")

    assert len(orders_df) == len(master_df)

    # --------------------------------------------------------
    # Uniqueness
    # --------------------------------------------------------

    print(
        f"Unique order IDs: "
        f"{orders_df['order_id'].nunique()}"
    )

    print(
        f"Unique transaction IDs: "
        f"{orders_df['transaction_id'].nunique()}"
    )

    assert orders_df["order_id"].is_unique
    assert orders_df["transaction_id"].is_unique

    # --------------------------------------------------------
    # Amount validation
    # --------------------------------------------------------

    assert (orders_df["order_amount"] > 0).all()

    print(
        f"Total order value: "
        f"₹{orders_df['order_amount'].sum():,.2f}"
    )

    # --------------------------------------------------------
    # Currency validation
    # --------------------------------------------------------

    assert (orders_df["currency"] == "INR").all()

    # --------------------------------------------------------
    # Verify against master dataset
    # --------------------------------------------------------

    master_sorted = master_df.sort_values(
        "transaction_id"
    ).reset_index(drop=True)

    orders_sorted = orders_df.sort_values(
        "transaction_id"
    ).reset_index(drop=True)

    columns_to_compare = [
        "order_id",
        "transaction_id",
        "merchant_id",
        "customer_id",
        "order_date",
        "order_amount",
        "currency",
        "payment_method",
    ]

    for column in columns_to_compare:

        assert (
            master_sorted[column].astype(str).values
            == orders_sorted[column].astype(str).values
        ).all(), f"Mismatch found in column: {column}"

    print("\nAll order records match the master dataset.")

    print("Orders validation successful.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("Generating orders.csv...")

    # Load master data
    master_df = load_master_transactions()

    # Generate orders
    orders_df = generate_orders(master_df)

    # Validate
    validate_orders(
        orders_df,
        master_df
    )

    # Save
    orders_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nOrders dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # Preview
    print("\nFirst 10 orders:")
    print(
        orders_df.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()