from pathlib import Path
import random

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent

CLEAN_DIR = BASE_DIR / "data" / "clean"
RAW_DIR = BASE_DIR / "data" / "raw"
GROUND_TRUTH_DIR = BASE_DIR / "data" / "ground_truth"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXCEPTION COUNTS
# ============================================================

EXCEPTION_COUNTS = {
    "AMOUNT_MISMATCH": 30,
    "MISSING_PAYMENT": 20,
    "MISSING_SETTLEMENT": 20,
    "DUPLICATE_PAYMENT": 15,
    "DUPLICATE_SETTLEMENT": 10,
    "FAILED_PAYMENT": 15,
    "PARTIAL_SETTLEMENT": 15,
    "REFUND": 15,
    "CHARGEBACK": 10,
    "SETTLEMENT_DELAY": 10,
    "INCORRECT_FEE": 10,
    "UNKNOWN_ADJUSTMENT": 5,
    "WRONG_TRANSACTION_REFERENCE": 10,
    "DATE_MISMATCH": 10,
    "MULTIPLE_PAYMENTS": 5,
}


# ============================================================
# LOAD CLEAN DATA
# ============================================================

def load_clean_data():

    files = {
        "orders": CLEAN_DIR / "orders.csv",
        "payments": CLEAN_DIR / "payments.csv",
        "bank": CLEAN_DIR / "bank_transactions.csv",
        "settlements": CLEAN_DIR / "settlements.csv",
    }

    for name, path in files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name}.csv not found at:\n{path}\n\n"
                "Make sure the clean datasets exist."
            )

    orders = pd.read_csv(files["orders"])
    payments = pd.read_csv(files["payments"])
    bank = pd.read_csv(files["bank"])
    settlements = pd.read_csv(files["settlements"])

    print("Clean datasets loaded successfully.")

    print(f"Orders: {len(orders)}")
    print(f"Payments: {len(payments)}")
    print(f"Bank transactions: {len(bank)}")
    print(f"Settlements: {len(settlements)}")

    return orders, payments, bank, settlements


# ============================================================
# SELECT TRANSACTIONS
# ============================================================

def select_exception_transactions(
    transaction_ids,
    counts
):

    shuffled = list(transaction_ids)

    random.shuffle(shuffled)

    total_required = sum(counts.values())

    if total_required > len(shuffled):
        raise ValueError(
            "Number of requested exceptions exceeds "
            "available transactions."
        )

    selected = {}

    start = 0

    for exception_type, count in counts.items():

        end = start + count

        selected[exception_type] = shuffled[
            start:end
        ]

        start = end

    return selected


# ============================================================
# EXCEPTION DETAILS STORAGE
# ============================================================

exception_details = []


def record_exception(
    transaction_id,
    exception_type,
    original_value=None,
    modified_value=None,
    difference=None
):

    exception_details.append({

        "transaction_id":
            transaction_id,

        "exception_type":
            exception_type,

        "original_value":
            original_value,

        "modified_value":
            modified_value,

        "difference":
            difference,
    })


# ============================================================
# AMOUNT MISMATCH
# ============================================================

def apply_amount_mismatch(
    settlements,
    transaction_ids
):

    for transaction_id in transaction_ids:

        index = settlements[
            settlements["transaction_id"]
            == transaction_id
        ].index

        if len(index) == 0:
            continue

        idx = index[0]

        original_amount = float(
            settlements.loc[
                idx,
                "net_amount"
            ]
        )

        difference = random.choice([
            100,
            250,
            500,
            750,
            1000,
        ])

        new_amount = max(
            0,
            original_amount - difference
        )

        settlements.loc[
            idx,
            "net_amount"
        ] = round(
            new_amount,
            2
        )

        record_exception(
            transaction_id,
            "AMOUNT_MISMATCH",
            original_amount,
            new_amount,
            round(
                original_amount - new_amount,
                2
            )
        )


# ============================================================
# MISSING PAYMENT
# ============================================================

def apply_missing_payment(
    payments,
    transaction_ids
):

    for transaction_id in transaction_ids:

        rows = payments[
            payments["transaction_id"]
            == transaction_id
        ]

        if rows.empty:
            continue

        payment_amount = float(
            rows.iloc[0]["payment_amount"]
        )

        payments.drop(
            rows.index,
            inplace=True
        )

        record_exception(
            transaction_id,
            "MISSING_PAYMENT",
            payment_amount,
            0.0,
            payment_amount
        )


# ============================================================
# MISSING SETTLEMENT
# ============================================================

def apply_missing_settlement(
    settlements,
    transaction_ids
):

    for transaction_id in transaction_ids:

        rows = settlements[
            settlements["transaction_id"]
            == transaction_id
        ]

        if rows.empty:
            continue

        net_amount = float(
            rows.iloc[0]["net_amount"]
        )

        settlements.drop(
            rows.index,
            inplace=True
        )

        record_exception(
            transaction_id,
            "MISSING_SETTLEMENT",
            net_amount,
            0.0,
            net_amount
        )


# ============================================================
# DUPLICATE PAYMENT
# ============================================================

def apply_duplicate_payment(
    payments,
    transaction_ids
):

    duplicates = []

    for transaction_id in transaction_ids:

        rows = payments[
            payments["transaction_id"]
            == transaction_id
        ]

        if rows.empty:
            continue

        row = rows.iloc[0].copy()

        original_amount = float(
            row["payment_amount"]
        )

        row["payment_id"] = (
            row["payment_id"] + "_DUP"
        )

        duplicates.append(row)

        record_exception(
            transaction_id,
            "DUPLICATE_PAYMENT",
            original_amount,
            original_amount * 2,
            original_amount
        )

    if duplicates:

        duplicate_df = pd.DataFrame(
            duplicates
        )

        payments = pd.concat(
            [
                payments,
                duplicate_df
            ],
            ignore_index=True
        )

    return payments


# ============================================================
# DUPLICATE SETTLEMENT
# ============================================================

def apply_duplicate_settlement(
    settlements,
    transaction_ids
):

    duplicates = []

    for transaction_id in transaction_ids:

        rows = settlements[
            settlements["transaction_id"]
            == transaction_id
        ]

        if rows.empty:
            continue

        row = rows.iloc[0].copy()

        original_net = float(
            row["net_amount"]
        )

        row["settlement_id"] = (
            row["settlement_id"] + "_DUP"
        )

        row["settlement_reference"] = (
            row["settlement_reference"] + "_DUP"
        )

        duplicates.append(row)

        record_exception(
            transaction_id,
            "DUPLICATE_SETTLEMENT",
            original_net,
            original_net * 2,
            original_net
        )

    if duplicates:

        duplicate_df = pd.DataFrame(
            duplicates
        )

        settlements = pd.concat(
            [
                settlements,
                duplicate_df
            ],
            ignore_index=True
        )

    return settlements


# ============================================================
# FAILED PAYMENT
# ============================================================

def apply_failed_payment(
    payments,
    transaction_ids
):

    for transaction_id in transaction_ids:

        index = payments[
            payments["transaction_id"]
            == transaction_id
        ].index

        if len(index) == 0:
            continue

        idx = index[0]

        original_status = (
            payments.loc[
                idx,
                "payment_status"
            ]
        )

        payment_amount = float(
            payments.loc[
                idx,
                "payment_amount"
            ]
        )

        payments.loc[
            idx,
            "payment_status"
        ] = "FAILED"

        record_exception(
            transaction_id,
            "FAILED_PAYMENT",
            original_status,
            "FAILED",
            payment_amount
        )


# ============================================================
# PARTIAL SETTLEMENT
# ============================================================

def apply_partial_settlement(
    settlements,
    transaction_ids
):

    for transaction_id in transaction_ids:

        index = settlements[
            settlements["transaction_id"]
            == transaction_id
        ].index

        if len(index) == 0:
            continue

        idx = index[0]

        gross = float(
            settlements.loc[
                idx,
                "gross_amount"
            ]
        )

        original_net = float(
            settlements.loc[
                idx,
                "net_amount"
            ]
        )

        partial_amount = round(
            gross * random.choice([
                0.50,
                0.60,
                0.70,
                0.80
            ]),
            2
        )

        settlements.loc[
            idx,
            "net_amount"
        ] = partial_amount

        record_exception(
            transaction_id,
            "PARTIAL_SETTLEMENT",
            original_net,
            partial_amount,
            round(
                original_net - partial_amount,
                2
            )
        )


# ============================================================
# REFUND
# ============================================================

def apply_refund(
    settlements,
    transaction_ids
):

    for transaction_id in transaction_ids:

        index = settlements[
            settlements["transaction_id"]
            == transaction_id
        ].index

        if len(index) == 0:
            continue

        idx = index[0]

        gross = float(
            settlements.loc[
                idx,
                "gross_amount"
            ]
        )

        original_net = float(
            settlements.loc[
                idx,
                "net_amount"
            ]
        )

        refund = round(
            gross * random.choice([
                0.10,
                0.20,
                0.25,
                0.30
            ]),
            2
        )

        settlements.loc[
            idx,
            "refund"
        ] = refund

        fee = float(
            settlements.loc[idx, "fee"]
        )

        tax = float(
            settlements.loc[idx, "tax"]
        )

        adjustment = float(
            settlements.loc[idx, "adjustment"]
        )

        chargeback = float(
            settlements.loc[idx, "chargeback"]
        )

        new_net = round(
            gross
            - fee
            - tax
            - refund
            - chargeback
            + adjustment,
            2
        )

        settlements.loc[
            idx,
            "net_amount"
        ] = new_net

        record_exception(
            transaction_id,
            "REFUND",
            original_net,
            new_net,
            refund
        )


# ============================================================
# CHARGEBACK
# ============================================================

def apply_chargeback(
    settlements,
    transaction_ids
):

    for transaction_id in transaction_ids:

        index = settlements[
            settlements["transaction_id"]
            == transaction_id
        ].index

        if len(index) == 0:
            continue

        idx = index[0]

        gross = float(
            settlements.loc[
                idx,
                "gross_amount"
            ]
        )

        original_net = float(
            settlements.loc[
                idx,
                "net_amount"
            ]
        )

        chargeback = round(
            gross * random.choice([
                0.50,
                0.75,
                1.00
            ]),
            2
        )

        settlements.loc[
            idx,
            "chargeback"
        ] = chargeback

        fee = float(
            settlements.loc[idx, "fee"]
        )

        tax = float(
            settlements.loc[idx, "tax"]
        )

        refund = float(
            settlements.loc[idx, "refund"]
        )

        adjustment = float(
            settlements.loc[idx, "adjustment"]
        )

        new_net = round(
            gross
            - fee
            - tax
            - refund
            - chargeback
            + adjustment,
            2
        )

        settlements.loc[
            idx,
            "net_amount"
        ] = new_net

        record_exception(
            transaction_id,
            "CHARGEBACK",
            original_net,
            new_net,
            chargeback
        )


# ============================================================
# SETTLEMENT DELAY
# ============================================================

def apply_settlement_delay(
    settlements,
    transaction_ids
):

    for transaction_id in transaction_ids:

        index = settlements[
            settlements["transaction_id"]
            == transaction_id
        ].index

        if len(index) == 0:
            continue

        idx = index[0]

        original_date = (
            settlements.loc[
                idx,
                "settlement_date"
            ]
        )

        current_date = pd.to_datetime(
            original_date
        )

        delay_days = random.choice([
            5,
            7,
            10,
            14
        ])

        delayed_date = (
            current_date
            + pd.Timedelta(
                days=delay_days
            )
        )

        new_date = delayed_date.strftime(
            "%Y-%m-%d"
        )

        settlements.loc[
            idx,
            "settlement_date"
        ] = new_date

        record_exception(
            transaction_id,
            "SETTLEMENT_DELAY",
            original_date,
            new_date,
            delay_days
        )


# ============================================================
# INCORRECT FEE
# ============================================================

def apply_incorrect_fee(
    settlements,
    transaction_ids
):

    for transaction_id in transaction_ids:

        index = settlements[
            settlements["transaction_id"]
            == transaction_id
        ].index

        if len(index) == 0:
            continue

        idx = index[0]

        gross = float(
            settlements.loc[
                idx,
                "gross_amount"
            ]
        )

        original_fee = float(
            settlements.loc[
                idx,
                "fee"
            ]
        )

        incorrect_fee = round(
            gross * random.choice([
                0.05,
                0.07,
                0.10
            ]),
            2
        )

        settlements.loc[
            idx,
            "fee"
        ] = incorrect_fee

        tax = float(
            settlements.loc[idx, "tax"]
        )

        refund = float(
            settlements.loc[idx, "refund"]
        )

        chargeback = float(
            settlements.loc[idx, "chargeback"]
        )

        adjustment = float(
            settlements.loc[idx, "adjustment"]
        )

        new_net = round(
            gross
            - incorrect_fee
            - tax
            - refund
            - chargeback
            + adjustment,
            2
        )

        settlements.loc[
            idx,
            "net_amount"
        ] = new_net

        record_exception(
            transaction_id,
            "INCORRECT_FEE",
            original_fee,
            incorrect_fee,
            round(
                incorrect_fee
                - original_fee,
                2
            )
        )


# ============================================================
# UNKNOWN ADJUSTMENT
# ============================================================

def apply_unknown_adjustment(
    settlements,
    transaction_ids
):

    for transaction_id in transaction_ids:

        index = settlements[
            settlements["transaction_id"]
            == transaction_id
        ].index

        if len(index) == 0:
            continue

        idx = index[0]

        original_adjustment = float(
            settlements.loc[
                idx,
                "adjustment"
            ]
        )

        adjustment = random.choice([
            -100,
            -250,
            -500,
            100,
            250
        ])

        settlements.loc[
            idx,
            "adjustment"
        ] = adjustment

        gross = float(
            settlements.loc[
                idx,
                "gross_amount"
            ]
        )

        fee = float(
            settlements.loc[
                idx,
                "fee"
            ]
        )

        tax = float(
            settlements.loc[
                idx,
                "tax"
            ]
        )

        refund = float(
            settlements.loc[
                idx,
                "refund"
            ]
        )

        chargeback = float(
            settlements.loc[
                idx,
                "chargeback"
            ]
        )

        new_net = round(
            gross
            - fee
            - tax
            - refund
            - chargeback
            + adjustment,
            2
        )

        settlements.loc[
            idx,
            "net_amount"
        ] = new_net

        record_exception(
            transaction_id,
            "UNKNOWN_ADJUSTMENT",
            original_adjustment,
            adjustment,
            adjustment - original_adjustment
        )


# ============================================================
# WRONG TRANSACTION REFERENCE
# ============================================================

def apply_wrong_transaction_reference(
    bank,
    transaction_ids,
    all_transaction_ids
):

    for transaction_id in transaction_ids:

        index = bank[
            bank["transaction_id"]
            == transaction_id
        ].index

        if len(index) == 0:
            continue

        idx = index[0]

        original_reference = (
            bank.loc[
                idx,
                "reference"
            ]
        )

        wrong_reference = random.choice(
            [
                tx
                for tx in all_transaction_ids
                if tx != transaction_id
            ]
        )

        bank.loc[
            idx,
            "reference"
        ] = wrong_reference

        record_exception(
            transaction_id,
            "WRONG_TRANSACTION_REFERENCE",
            original_reference,
            wrong_reference,
            None
        )


# ============================================================
# DATE MISMATCH
# ============================================================

def apply_date_mismatch(
    payments,
    transaction_ids
):

    for transaction_id in transaction_ids:

        index = payments[
            payments["transaction_id"]
            == transaction_id
        ].index

        if len(index) == 0:
            continue

        idx = index[0]

        original_date = (
            payments.loc[
                idx,
                "payment_date"
            ]
        )

        current_date = pd.to_datetime(
            original_date
        )

        new_date = (
            current_date
            + pd.Timedelta(days=10)
        ).strftime(
            "%Y-%m-%d"
        )

        payments.loc[
            idx,
            "payment_date"
        ] = new_date

        record_exception(
            transaction_id,
            "DATE_MISMATCH",
            original_date,
            new_date,
            10
        )


# ============================================================
# MULTIPLE PAYMENTS
# ============================================================

def apply_multiple_payments(
    payments,
    transaction_ids
):

    additional_payments = []

    for transaction_id in transaction_ids:

        rows = payments[
            payments["transaction_id"]
            == transaction_id
        ]

        if rows.empty:
            continue

        row = rows.iloc[0].copy()

        original_amount = float(
            row["payment_amount"]
        )

        additional_amount = round(
            original_amount * 0.30,
            2
        )

        row["payment_id"] = (
            row["payment_id"]
            + "_PART2"
        )

        row["payment_amount"] = (
            additional_amount
        )

        additional_payments.append(
            row
        )

        record_exception(
            transaction_id,
            "MULTIPLE_PAYMENTS",
            original_amount,
            original_amount
            + additional_amount,
            additional_amount
        )

    if additional_payments:

        additional_df = pd.DataFrame(
            additional_payments
        )

        payments = pd.concat(
            [
                payments,
                additional_df
            ],
            ignore_index=True
        )

    return payments


# ============================================================
# SAVE DATASETS
# ============================================================

def save_datasets(
    orders,
    payments,
    bank,
    settlements
):

    orders.to_csv(
        RAW_DIR / "orders.csv",
        index=False
    )

    payments.to_csv(
        RAW_DIR / "payments.csv",
        index=False
    )

    bank.to_csv(
        RAW_DIR / "bank_transactions.csv",
        index=False
    )

    settlements.to_csv(
        RAW_DIR / "settlements.csv",
        index=False
    )


# ============================================================
# SAVE EXCEPTION MAPPING
# ============================================================

def save_exception_mapping(
    selected
):

    mapping_rows = []

    for exception_type, transaction_ids in selected.items():

        for transaction_id in transaction_ids:

            mapping_rows.append({
                "transaction_id":
                    transaction_id,

                "exception_type":
                    exception_type
            })

    mapping_df = pd.DataFrame(
        mapping_rows
    )

    output_file = (
        GROUND_TRUTH_DIR
        / "exception_mapping.csv"
    )

    mapping_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nException mapping saved to:\n"
        f"{output_file}"
    )

    print(
        f"Total exception mappings: "
        f"{len(mapping_df)}"
    )

    return mapping_df


# ============================================================
# SAVE EXCEPTION DETAILS
# ============================================================

def save_exception_details():

    details_df = pd.DataFrame(
        exception_details
    )

    output_file = (
        GROUND_TRUTH_DIR
        / "exception_details.csv"
    )

    details_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nException details saved to:\n"
        f"{output_file}"
    )

    print(
        f"Total exception details: "
        f"{len(details_df)}"
    )

    return details_df


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Starting exception injection..."
    )

    # --------------------------------------------------------
    # Load clean datasets
    # --------------------------------------------------------

    orders, payments, bank, settlements = (
        load_clean_data()
    )

    # --------------------------------------------------------
    # Get transaction IDs
    # --------------------------------------------------------

    transaction_ids = (
        orders["transaction_id"]
        .tolist()
    )

    # --------------------------------------------------------
    # Select exceptions
    # --------------------------------------------------------

    selected = (
        select_exception_transactions(
            transaction_ids,
            EXCEPTION_COUNTS
        )
    )

    # --------------------------------------------------------
    # Reset exception details
    # --------------------------------------------------------

    exception_details.clear()

    # --------------------------------------------------------
    # Save mapping
    # --------------------------------------------------------

    save_exception_mapping(
        selected
    )

    # --------------------------------------------------------
    # Apply exceptions
    # --------------------------------------------------------

    apply_amount_mismatch(
        settlements,
        selected["AMOUNT_MISMATCH"]
    )

    apply_missing_payment(
        payments,
        selected["MISSING_PAYMENT"]
    )

    apply_missing_settlement(
        settlements,
        selected["MISSING_SETTLEMENT"]
    )

    payments = apply_duplicate_payment(
        payments,
        selected["DUPLICATE_PAYMENT"]
    )

    settlements = apply_duplicate_settlement(
        settlements,
        selected["DUPLICATE_SETTLEMENT"]
    )

    apply_failed_payment(
        payments,
        selected["FAILED_PAYMENT"]
    )

    apply_partial_settlement(
        settlements,
        selected["PARTIAL_SETTLEMENT"]
    )

    apply_refund(
        settlements,
        selected["REFUND"]
    )

    apply_chargeback(
        settlements,
        selected["CHARGEBACK"]
    )

    apply_settlement_delay(
        settlements,
        selected["SETTLEMENT_DELAY"]
    )

    apply_incorrect_fee(
        settlements,
        selected["INCORRECT_FEE"]
    )

    apply_unknown_adjustment(
        settlements,
        selected["UNKNOWN_ADJUSTMENT"]
    )

    apply_wrong_transaction_reference(
        bank,
        selected["WRONG_TRANSACTION_REFERENCE"],
        transaction_ids
    )

    apply_date_mismatch(
        payments,
        selected["DATE_MISMATCH"]
    )

    payments = apply_multiple_payments(
        payments,
        selected["MULTIPLE_PAYMENTS"]
    )

    # --------------------------------------------------------
    # Save modified datasets
    # --------------------------------------------------------

    save_datasets(
        orders,
        payments,
        bank,
        settlements
    )

    # --------------------------------------------------------
    # Save detailed ground-truth information
    # --------------------------------------------------------

    save_exception_details()

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n========== EXCEPTION SUMMARY =========="
    )

    for exception_type, ids in selected.items():

        print(
            f"{exception_type}: "
            f"{len(ids)}"
        )

    print(
        "\n========== FINAL DATASET SIZES =========="
    )

    print(
        f"Orders: {len(orders)}"
    )

    print(
        f"Payments: {len(payments)}"
    )

    print(
        f"Bank transactions: {len(bank)}"
    )

    print(
        f"Settlements: {len(settlements)}"
    )

    print(
        "\nException injection completed."
    )


if __name__ == "__main__":
    main()