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
OUTPUT_FILE = OUTPUT_DIR / "settlements.csv"

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

    print(
        f"Loaded master transactions: {len(df)} rows"
    )

    return df


# ============================================================
# GENERATE SETTLEMENT DATE
# ============================================================

def generate_settlement_date(order_date):

    """
    Generate a settlement date.

    For the clean baseline dataset, settlement occurs
    between 1 and 3 days after the order.
    """

    order_date = datetime.strptime(
        order_date,
        "%Y-%m-%d"
    )

    delay_days = random.choice([
        1,
        1,
        1,
        2,
        2,
        3
    ])

    settlement_date = (
        order_date
        + timedelta(days=delay_days)
    )

    return settlement_date.strftime(
        "%Y-%m-%d"
    )


# ============================================================
# CALCULATE PROCESSING FEE
# ============================================================

def calculate_fee(amount):

    """
    Generate a realistic processing fee.

    We use a small percentage of the transaction amount,
    rounded to two decimal places.
    """

    fee_rate = random.choice([
        0.010,
        0.012,
        0.015,
        0.018,
        0.020
    ])

    fee = round(
        amount * fee_rate,
        2
    )

    return fee


# ============================================================
# CALCULATE TAX
# ============================================================

def calculate_tax(fee):

    """
    Calculate tax on the processing fee.

    Using 18% as a synthetic GST-like tax rate.
    """

    tax_rate = 0.18

    tax = round(
        fee * tax_rate,
        2
    )

    return tax


# ============================================================
# GENERATE SETTLEMENTS
# ============================================================

def generate_settlements(master_df):

    settlements = []

    for _, transaction in master_df.iterrows():

        transaction_id = (
            transaction["transaction_id"]
        )

        order_amount = float(
            transaction["order_amount"]
        )

        settlement_id = (
            f"SET{transaction_id[3:]}"
        )

        settlement_date = (
            generate_settlement_date(
                transaction["order_date"]
            )
        )

        # Calculate normal fee
        fee = calculate_fee(
            order_amount
        )

        # Calculate tax on fee
        tax = calculate_tax(
            fee
        )

        # Clean baseline:
        # no refund
        # no chargeback
        # no adjustment
        refund = 0.00
        chargeback = 0.00
        adjustment = 0.00

        # Calculate net settlement
        net_amount = round(
            order_amount
            - fee
            - tax
            - refund
            - chargeback
            + adjustment,
            2
        )

        settlements.append({

            "settlement_id":
                settlement_id,

            "transaction_id":
                transaction_id,

            "merchant_id":
                transaction["merchant_id"],

            "settlement_date":
                settlement_date,

            "gross_amount":
                order_amount,

            "fee":
                fee,

            "tax":
                tax,

            "adjustment":
                adjustment,

            "refund":
                refund,

            "chargeback":
                chargeback,

            "net_amount":
                net_amount,

            "settlement_status":
                "SETTLED",

            "settlement_reference":
                settlement_id,

            "currency":
                transaction["currency"],
        })

    return pd.DataFrame(
        settlements
    )


# ============================================================
# VALIDATE SETTLEMENTS
# ============================================================

def validate_settlements(
    settlements_df,
    master_df
):

    print(
        "\n========== SETTLEMENTS VALIDATION =========="
    )

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    print(
        f"Total settlements: "
        f"{len(settlements_df)}"
    )

    assert (
        len(settlements_df)
        == len(master_df)
    )

    # --------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------

    print(
        f"Unique settlement IDs: "
        f"{settlements_df['settlement_id'].nunique()}"
    )

    print(
        f"Unique transaction IDs: "
        f"{settlements_df['transaction_id'].nunique()}"
    )

    assert (
        settlements_df[
            "settlement_id"
        ].is_unique
    )

    assert (
        settlements_df[
            "transaction_id"
        ].is_unique
    )

    # --------------------------------------------------------
    # Settlement status
    # --------------------------------------------------------

    print("\nSettlement status:")

    print(
        settlements_df[
            "settlement_status"
        ].value_counts()
    )

    assert (
        settlements_df[
            "settlement_status"
        ] == "SETTLED"
    ).all()

    # --------------------------------------------------------
    # Amount validation
    # --------------------------------------------------------

    amount_columns = [
        "gross_amount",
        "fee",
        "tax",
        "adjustment",
        "refund",
        "chargeback",
        "net_amount",
    ]

    for column in amount_columns:

        assert (
            settlements_df[column]
            >= 0
        ).all(), (
            f"Negative value found "
            f"in {column}"
        )

    # --------------------------------------------------------
    # Verify net amount calculation
    # --------------------------------------------------------

    calculated_net = (
        settlements_df["gross_amount"]
        - settlements_df["fee"]
        - settlements_df["tax"]
        - settlements_df["refund"]
        - settlements_df["chargeback"]
        + settlements_df["adjustment"]
    ).round(2)

    assert (
        calculated_net
        == settlements_df["net_amount"]
    ).all()

    print(
        "\nAll net settlement calculations "
        "are correct."
    )

    # --------------------------------------------------------
    # Compare gross amount with master
    # --------------------------------------------------------

    merged = settlements_df.merge(
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
            "_settlement",
            "_master"
        ),
        validate="one_to_one",
    )

    # Every transaction must exist
    assert (
        merged["order_amount"].notna()
    ).all()

    # Gross amount must equal order amount
    assert (
        merged["gross_amount"]
        == merged["order_amount"]
    ).all()

    # Merchant must match
    assert (
        merged["merchant_id_settlement"]
        == merged["merchant_id_master"]
    ).all()

    # Currency must match
    assert (
        merged["currency_settlement"]
        == merged["currency_master"]
    ).all()

    # --------------------------------------------------------
    # Print totals
    # --------------------------------------------------------

    print(
        f"\nTotal gross amount: "
        f"₹{settlements_df['gross_amount'].sum():,.2f}"
    )

    print(
        f"Total fees: "
        f"₹{settlements_df['fee'].sum():,.2f}"
    )

    print(
        f"Total tax: "
        f"₹{settlements_df['tax'].sum():,.2f}"
    )

    print(
        f"Total net settlement: "
        f"₹{settlements_df['net_amount'].sum():,.2f}"
    )

    print(
        "\nSettlements validation successful."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Generating settlements.csv..."
    )

    master_df = (
        load_master_transactions()
    )

    settlements_df = (
        generate_settlements(
            master_df
        )
    )

    validate_settlements(
        settlements_df,
        master_df
    )

    settlements_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSettlements dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nFirst 10 settlements:"
    )

    print(
        settlements_df.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()