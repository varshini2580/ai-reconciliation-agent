from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
RECON_DIR = DATA_DIR / "reconciliation"


# ============================================================
# LOAD DATA
# ============================================================

def load(name, directory):

    path = directory / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    return pd.read_csv(path)


def main():

    print("=" * 70)
    print("        PHASE 5.1 — ML DATA INSPECTION")
    print("=" * 70)

    orders = load("orders", RAW_DIR)
    payments = load("payments", RAW_DIR)
    bank = load("bank_transactions", RAW_DIR)
    settlements = load("settlements", RAW_DIR)

    ground_truth = load(
        "ground_truth",
        GROUND_TRUTH_DIR
    )

    exception_details = load(
        "exception_details",
        GROUND_TRUTH_DIR
    )

    print("\n========== DATASET SIZES ==========")

    print("Orders:", orders.shape)
    print("Payments:", payments.shape)
    print("Bank:", bank.shape)
    print("Settlements:", settlements.shape)
    print("Ground truth:", ground_truth.shape)
    print("Exception details:", exception_details.shape)

    # --------------------------------------------------------
    # RAW COLUMNS
    # --------------------------------------------------------

    print("\n========== RAW DATA COLUMNS ==========")

    for name, df in [
        ("orders", orders),
        ("payments", payments),
        ("bank_transactions", bank),
        ("settlements", settlements),
    ]:

        print(f"\n{name}:")
        print(list(df.columns))

    # --------------------------------------------------------
    # GROUND TRUTH COLUMNS
    # --------------------------------------------------------

    print("\n========== GROUND TRUTH COLUMNS ==========")

    print("ground_truth:")
    print(list(ground_truth.columns))

    print("\nexception_details:")
    print(list(exception_details.columns))

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    print("\n========== DATA TYPES ==========")

    for name, df in [
        ("orders", orders),
        ("payments", payments),
        ("bank_transactions", bank),
        ("settlements", settlements),
    ]:

        print(f"\n{name}:")
        print(df.dtypes.to_string())

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    print("\n========== MISSING VALUES ==========")

    for name, df in [
        ("orders", orders),
        ("payments", payments),
        ("bank_transactions", bank),
        ("settlements", settlements),
    ]:

        print(f"\n{name}:")
        missing = df.isna().sum()
        print(
            missing[missing > 0].to_string()
            if (missing > 0).any()
            else "No missing values"
        )

    # --------------------------------------------------------
    # CATEGORICAL VALUES
    # --------------------------------------------------------

    print("\n========== CATEGORICAL VALUES ==========")

    categorical_columns = [
        ("payments", payments, "payment_status"),
        ("payments", payments, "payment_method"),
        ("bank_transactions", bank, "transaction_type"),
        ("settlements", settlements, "settlement_status"),
    ]

    for name, df, column in categorical_columns:

        print(f"\n{name}.{column}:")

        if column in df.columns:
            print(
                df[column]
                .value_counts(dropna=False)
                .to_string()
            )

    # --------------------------------------------------------
    # GROUND TRUTH DISTRIBUTION
    # --------------------------------------------------------

    print("\n========== TARGET DISTRIBUTION ==========")

    target_column = "true_exception_type"

    if target_column in ground_truth.columns:

        print(
            ground_truth[target_column]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # --------------------------------------------------------
    # EXCEPTION DETAILS
    # --------------------------------------------------------

    print("\n========== EXCEPTION DETAILS SAMPLE ==========")

    print(
        exception_details
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # EXISTING MATCH TABLE
    # --------------------------------------------------------

    match_file = RECON_DIR / "match_table.csv"

    if match_file.exists():

        match = pd.read_csv(match_file)

        print("\n========== MATCH TABLE ==========")

        print("Shape:", match.shape)

        print("Columns:")
        print(list(match.columns))

        print("\nData types:")
        print(match.dtypes.to_string())

        print("\nSample:")
        print(
            match.head(10)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # RECONCILIATION RESULTS
    # --------------------------------------------------------

    results_file = (
        RECON_DIR /
        "reconciliation_results.csv"
    )

    if results_file.exists():

        results = pd.read_csv(results_file)

        print("\n========== RECONCILIATION RESULTS ==========")

        print("Shape:", results.shape)

        print("Columns:")
        print(list(results.columns))

    # --------------------------------------------------------
    # LEAKAGE WARNING
    # --------------------------------------------------------

    print("\n========== ML TARGET / LEAKAGE CHECK ==========")

    forbidden = {
        "true_exception_type",
        "exception_type",
        "status",
        "ground_truth",
    }

    print(
        "These fields must NOT be used as ML input features:"
    )

    for column in sorted(forbidden):
        print(" -", column)

    print("\nPhase 5.1 inspection completed.")


if __name__ == "__main__":
    main()