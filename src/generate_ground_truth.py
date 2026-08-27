from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MASTER_FILE = (
    BASE_DIR
    / "data"
    / "master_transactions.csv"
)

RAW_DIR = (
    BASE_DIR
    / "data"
    / "raw"
)

GROUND_TRUTH_DIR = (
    BASE_DIR
    / "data"
    / "ground_truth"
)

MAPPING_FILE = (
    GROUND_TRUTH_DIR
    / "exception_mapping.csv"
)

DETAILS_FILE = (
    GROUND_TRUTH_DIR
    / "exception_details.csv"
)

OUTPUT_FILE = (
    GROUND_TRUTH_DIR
    / "ground_truth.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    required_files = {
        "master": MASTER_FILE,
        "orders": RAW_DIR / "orders.csv",
        "payments": RAW_DIR / "payments.csv",
        "settlements": RAW_DIR / "settlements.csv",
        "mapping": MAPPING_FILE,
        "details": DETAILS_FILE,
    }

    for name, path in required_files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

    master = pd.read_csv(
        MASTER_FILE
    )

    orders = pd.read_csv(
        RAW_DIR / "orders.csv"
    )

    payments = pd.read_csv(
        RAW_DIR / "payments.csv"
    )

    settlements = pd.read_csv(
        RAW_DIR / "settlements.csv"
    )

    mapping = pd.read_csv(
        MAPPING_FILE
    )

    details = pd.read_csv(
        DETAILS_FILE
    )

    return (
        master,
        orders,
        payments,
        settlements,
        mapping,
        details,
    )


# ============================================================
# BUILD ID MAP
# ============================================================

def build_id_map(
    dataframe,
    transaction_column,
    id_column
):

    result = {}

    for transaction_id, group in (
        dataframe.groupby(transaction_column)
    ):

        ids = (
            group[id_column]
            .astype(str)
            .tolist()
        )

        result[transaction_id] = (
            "|".join(ids)
        )

    return result


# ============================================================
# BUILD EXCEPTION MAP
# ============================================================

def build_exception_map(
    mapping
):

    result = {}

    for _, row in mapping.iterrows():

        result[
            row["transaction_id"]
        ] = row["exception_type"]

    return result


# ============================================================
# BUILD DIFFERENCE MAP
# ============================================================

def build_difference_map(
    details
):

    result = {}

    for _, row in details.iterrows():

        difference = row["difference"]

        if pd.isna(difference):

            result[
                row["transaction_id"]
            ] = None

        else:

            result[
                row["transaction_id"]
            ] = difference

    return result


# ============================================================
# BUILD GROUND TRUTH
# ============================================================

def build_ground_truth(
    master,
    payments,
    settlements,
    mapping,
    details
):

    payment_map = build_id_map(
        payments,
        "transaction_id",
        "payment_id"
    )

    settlement_map = build_id_map(
        settlements,
        "transaction_id",
        "settlement_id"
    )

    exception_map = build_exception_map(
        mapping
    )

    difference_map = build_difference_map(
        details
    )

    rows = []

    for _, master_row in master.iterrows():

        transaction_id = (
            master_row["transaction_id"]
        )

        order_id = (
            master_row["order_id"]
        )

        payment_id = payment_map.get(
            transaction_id,
            None
        )

        settlement_id = settlement_map.get(
            transaction_id,
            None
        )

        exception_type = (
            exception_map.get(
                transaction_id,
                "NONE"
            )
        )

        if exception_type == "NONE":

            true_status = "MATCHED"

            difference = 0.0

        else:

            true_status = "EXCEPTION"

            difference = difference_map.get(
                transaction_id,
                None
            )

        rows.append({

            "transaction_id":
                transaction_id,

            "order_id":
                order_id,

            "payment_id":
                payment_id,

            "settlement_id":
                settlement_id,

            "true_status":
                true_status,

            "true_exception_type":
                exception_type,

            "true_difference":
                difference,
        })

    return pd.DataFrame(rows)


# ============================================================
# VALIDATION
# ============================================================

def validate_ground_truth(
    ground_truth,
    master,
    mapping,
    details
):

    print(
        "\n========== GROUND TRUTH VALIDATION =========="
    )

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    assert (
        len(ground_truth)
        == len(master)
    )

    print(
        f"Ground truth rows: "
        f"{len(ground_truth)}"
    )

    # --------------------------------------------------------
    # Unique transaction IDs
    # --------------------------------------------------------

    assert (
        ground_truth[
            "transaction_id"
        ].is_unique
    )

    print(
        f"Unique transaction IDs: "
        f"{ground_truth['transaction_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Exception count
    # --------------------------------------------------------

    expected_exceptions = len(
        mapping
    )

    actual_exceptions = (
        ground_truth[
            ground_truth[
                "true_status"
            ] == "EXCEPTION"
        ].shape[0]
    )

    assert (
        expected_exceptions
        == actual_exceptions
    )

    print(
        f"Expected exceptions: "
        f"{expected_exceptions}"
    )

    print(
        f"Actual exceptions: "
        f"{actual_exceptions}"
    )

    # --------------------------------------------------------
    # Normal count
    # --------------------------------------------------------

    normal_count = (
        ground_truth[
            ground_truth[
                "true_status"
            ] == "MATCHED"
        ].shape[0]
    )

    print(
        f"Normal transactions: "
        f"{normal_count}"
    )

    assert (
        normal_count
        + actual_exceptions
        == len(master)
    )

    # --------------------------------------------------------
    # Exception type counts
    # --------------------------------------------------------

    print(
        "\nTrue exception types:"
    )

    print(
        ground_truth[
            "true_exception_type"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Details count
    # --------------------------------------------------------

    assert (
        len(details)
        == expected_exceptions
    )

    print(
        f"\nException detail records: "
        f"{len(details)}"
    )

    # --------------------------------------------------------
    # Difference count
    # --------------------------------------------------------

    known_difference_count = (
        ground_truth[
            "true_difference"
        ].notna()
        .sum()
    )

    print(
        f"Ground-truth records with "
        f"known difference: "
        f"{known_difference_count}"
    )

    print(
        "\nGround truth validation successful."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Generating ground_truth.csv..."
    )

    (
        master,
        orders,
        payments,
        settlements,
        mapping,
        details,
    ) = load_data()

    ground_truth = build_ground_truth(
        master,
        payments,
        settlements,
        mapping,
        details
    )

    validate_ground_truth(
        ground_truth,
        master,
        mapping,
        details
    )

    ground_truth.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nGround truth saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nFirst 10 ground truth records:"
    )

    print(
        ground_truth.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()