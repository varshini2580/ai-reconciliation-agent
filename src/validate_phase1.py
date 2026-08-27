from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MASTER_FILE = DATA_DIR / "master_transactions.csv"

CLEAN_DIR = DATA_DIR / "clean"
RAW_DIR = DATA_DIR / "raw"
GT_DIR = DATA_DIR / "ground_truth"


# ============================================================
# LOAD FILE
# ============================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(path)


# ============================================================
# VALIDATE FILES
# ============================================================

def validate_files():

    print(
        "\n========== FILE VALIDATION =========="
    )

    files = [
        MASTER_FILE,

        CLEAN_DIR / "orders.csv",
        CLEAN_DIR / "payments.csv",
        CLEAN_DIR / "bank_transactions.csv",
        CLEAN_DIR / "settlements.csv",

        RAW_DIR / "orders.csv",
        RAW_DIR / "payments.csv",
        RAW_DIR / "bank_transactions.csv",
        RAW_DIR / "settlements.csv",

        GT_DIR / "exception_mapping.csv",
        GT_DIR / "exception_details.csv",
        GT_DIR / "ground_truth.csv",
    ]

    for file in files:

        assert file.exists(), (
            f"Missing file: {file}"
        )

        print(
            f"[OK] {file.relative_to(BASE_DIR)}"
        )


# ============================================================
# VALIDATE MASTER
# ============================================================

def validate_master(master):

    print(
        "\n========== MASTER VALIDATION =========="
    )

    assert len(master) == 1000

    assert (
        master["transaction_id"].is_unique
    )

    assert (
        master["order_id"].is_unique
    )

    print(
        f"Transactions: {len(master)}"
    )

    print(
        "Unique transaction IDs: 1000"
    )

    print(
        "Unique order IDs: 1000"
    )


# ============================================================
# VALIDATE CLEAN DATA
# ============================================================

def validate_clean_data(
    master,
    orders,
    payments,
    bank,
    settlements
):

    print(
        "\n========== CLEAN DATA VALIDATION =========="
    )

    assert len(orders) == 1000
    assert len(payments) == 1000
    assert len(bank) == 1000
    assert len(settlements) == 1000

    assert (
        orders["transaction_id"].is_unique
    )

    assert (
        payments["transaction_id"].is_unique
    )

    assert (
        bank["transaction_id"].is_unique
    )

    assert (
        settlements["transaction_id"].is_unique
    )

    master_ids = set(
        master["transaction_id"]
    )

    assert (
        set(orders["transaction_id"])
        == master_ids
    )

    assert (
        set(payments["transaction_id"])
        == master_ids
    )

    assert (
        set(bank["transaction_id"])
        == master_ids
    )

    assert (
        set(settlements["transaction_id"])
        == master_ids
    )

    print(
        "Orders: 1000"
    )

    print(
        "Payments: 1000"
    )

    print(
        "Bank transactions: 1000"
    )

    print(
        "Settlements: 1000"
    )

    print(
        "All clean relationships valid."
    )


# ============================================================
# VALIDATE RAW DATA
# ============================================================

def validate_raw_data(
    orders,
    payments,
    bank,
    settlements
):

    print(
        "\n========== RAW DATA VALIDATION =========="
    )

    assert len(orders) == 1000
    assert len(payments) == 1000
    assert len(bank) == 1000
    assert len(settlements) == 990

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

    # Payment duplicates are expected
    payment_duplicate_count = (
        payments["transaction_id"]
        .duplicated()
        .sum()
    )

    # Settlement duplicates are expected
    settlement_duplicate_count = (
        settlements["transaction_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Payment duplicate transaction rows: "
        f"{payment_duplicate_count}"
    )

    print(
        f"Settlement duplicate transaction rows: "
        f"{settlement_duplicate_count}"
    )


# ============================================================
# VALIDATE GROUND TRUTH
# ============================================================

def validate_ground_truth(
    master,
    mapping,
    details,
    ground_truth
):

    print(
        "\n========== GROUND TRUTH VALIDATION =========="
    )

    assert len(ground_truth) == 1000

    assert (
        ground_truth["transaction_id"]
        .is_unique
    )

    assert len(mapping) == 200

    assert len(details) == 200

    exception_count = (
        ground_truth[
            ground_truth["true_status"]
            == "EXCEPTION"
        ].shape[0]
    )

    matched_count = (
        ground_truth[
            ground_truth["true_status"]
            == "MATCHED"
        ].shape[0]
    )

    assert exception_count == 200

    assert matched_count == 800

    assert (
        exception_count
        + matched_count
        == len(master)
    )

    print(
        f"Ground truth rows: "
        f"{len(ground_truth)}"
    )

    print(
        f"Matched: "
        f"{matched_count}"
    )

    print(
        f"Exceptions: "
        f"{exception_count}"
    )

    print(
        f"Exception mappings: "
        f"{len(mapping)}"
    )

    print(
        f"Exception details: "
        f"{len(details)}"
    )


# ============================================================
# VALIDATE EXCEPTION TYPES
# ============================================================

def validate_exception_types(
    mapping,
    ground_truth
):

    print(
        "\n========== EXCEPTION TYPE VALIDATION =========="
    )

    mapping_counts = (
        mapping[
            "exception_type"
        ]
        .value_counts()
        .sort_index()
    )

    ground_truth_counts = (
        ground_truth[
            ground_truth[
                "true_exception_type"
            ] != "NONE"
        ][
            "true_exception_type"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nMapping counts:"
    )

    print(mapping_counts)

    print(
        "\nGround truth counts:"
    )

    print(ground_truth_counts)

    assert (
        mapping_counts.equals(
            ground_truth_counts
        )
    )

    print(
        "\nException type counts match."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "================================================"
    )

    print(
        "        PHASE 1 FINAL VALIDATION"
    )

    print(
        "================================================"
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    validate_files()

    # --------------------------------------------------------
    # Load master
    # --------------------------------------------------------

    master = load_csv(
        MASTER_FILE
    )

    # --------------------------------------------------------
    # Load clean
    # --------------------------------------------------------

    clean_orders = load_csv(
        CLEAN_DIR / "orders.csv"
    )

    clean_payments = load_csv(
        CLEAN_DIR / "payments.csv"
    )

    clean_bank = load_csv(
        CLEAN_DIR / "bank_transactions.csv"
    )

    clean_settlements = load_csv(
        CLEAN_DIR / "settlements.csv"
    )

    # --------------------------------------------------------
    # Load raw
    # --------------------------------------------------------

    raw_orders = load_csv(
        RAW_DIR / "orders.csv"
    )

    raw_payments = load_csv(
        RAW_DIR / "payments.csv"
    )

    raw_bank = load_csv(
        RAW_DIR / "bank_transactions.csv"
    )

    raw_settlements = load_csv(
        RAW_DIR / "settlements.csv"
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    mapping = load_csv(
        GT_DIR / "exception_mapping.csv"
    )

    details = load_csv(
        GT_DIR / "exception_details.csv"
    )

    ground_truth = load_csv(
        GT_DIR / "ground_truth.csv"
    )

    # --------------------------------------------------------
    # Run validations
    # --------------------------------------------------------

    validate_master(
        master
    )

    validate_clean_data(
        master,
        clean_orders,
        clean_payments,
        clean_bank,
        clean_settlements
    )

    validate_raw_data(
        raw_orders,
        raw_payments,
        raw_bank,
        raw_settlements
    )

    validate_ground_truth(
        master,
        mapping,
        details,
        ground_truth
    )

    validate_exception_types(
        mapping,
        ground_truth
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "\n================================================"
    )

    print(
        "       PHASE 1 VALIDATION SUCCESSFUL"
    )

    print(
        "================================================"
    )

    print(
        "\nAll datasets are ready for Phase 2."
    )


if __name__ == "__main__":
    main()