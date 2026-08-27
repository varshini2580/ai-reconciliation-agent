from pathlib import Path
import sys

import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RECON_DIR = DATA_DIR / "reconciliation"

INPUT_FILE = RECON_DIR / "ml_dataset.csv"

BINARY_TRAIN_FILE = RECON_DIR / "ml_binary_train.csv"
BINARY_TEST_FILE = RECON_DIR / "ml_binary_test.csv"

MULTICLASS_TRAIN_FILE = RECON_DIR / "ml_multiclass_train.csv"
MULTICLASS_TEST_FILE = RECON_DIR / "ml_multiclass_test.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 70)
    print("        PHASE 5.3 — TRAIN / TEST SPLIT")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"ML dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"ML dataset records: {len(df)}")

    required = [
        "transaction_id",
        "target_exception",
        "target_exception_type",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("[OK] Input schema valid")

    return df


# ============================================================
# BINARY SPLIT
# ============================================================

def create_binary_split(df):

    print()
    print("========== BINARY SPLIT ==========")

    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target_exception"],
    )

    train = train.copy()
    test = test.copy()

    print(f"Training records: {len(train)}")
    print(f"Testing records:  {len(test)}")

    print("\nTraining target distribution:")
    print(
        train["target_exception"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTesting target distribution:")
    print(
        test["target_exception"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return train, test


# ============================================================
# MULTICLASS SPLIT
# ============================================================

def create_multiclass_split(df):

    print()
    print("========== MULTICLASS SPLIT ==========")

    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target_exception_type"],
    )

    train = train.copy()
    test = test.copy()

    print(f"Training records: {len(train)}")
    print(f"Testing records:  {len(test)}")

    print("\nTraining exception distribution:")
    print(
        train["target_exception_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTesting exception distribution:")
    print(
        test["target_exception_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return train, test


# ============================================================
# VALIDATION
# ============================================================

def validate_split(
    original,
    train,
    test,
    target_column,
    split_name,
):

    print()
    print(
        f"========== {split_name} VALIDATION =========="
    )

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    if len(train) + len(test) != len(original):

        raise ValueError(
            f"{split_name}: train + test count mismatch."
        )

    print("[OK] Record counts preserved")

    # --------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------

    train_ids = set(
        train["transaction_id"]
    )

    test_ids = set(
        test["transaction_id"]
    )

    if train_ids.intersection(test_ids):

        raise ValueError(
            f"{split_name}: transaction leakage detected "
            "between train and test."
        )

    print("[OK] No transaction overlap")

    # --------------------------------------------------------
    # All records preserved
    # --------------------------------------------------------

    combined_ids = train_ids | test_ids
    original_ids = set(
        original["transaction_id"]
    )

    if combined_ids != original_ids:

        raise ValueError(
            f"{split_name}: train/test IDs do not "
            "cover exactly the original dataset."
        )

    print("[OK] All transactions preserved")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print(
        f"[OK] Stratified target: {target_column}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        df = load_dataset()

        # ----------------------------------------------------
        # BINARY
        # ----------------------------------------------------

        binary_train, binary_test = (
            create_binary_split(df)
        )

        validate_split(
            df,
            binary_train,
            binary_test,
            "target_exception",
            "BINARY",
        )

        # ----------------------------------------------------
        # MULTICLASS
        # ----------------------------------------------------

        multiclass_train, multiclass_test = (
            create_multiclass_split(df)
        )

        validate_split(
            df,
            multiclass_train,
            multiclass_test,
            "target_exception_type",
            "MULTICLASS",
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        binary_train.to_csv(
            BINARY_TRAIN_FILE,
            index=False,
        )

        binary_test.to_csv(
            BINARY_TEST_FILE,
            index=False,
        )

        multiclass_train.to_csv(
            MULTICLASS_TRAIN_FILE,
            index=False,
        )

        multiclass_test.to_csv(
            MULTICLASS_TEST_FILE,
            index=False,
        )

        print()
        print("=" * 70)
        print("TRAIN / TEST FILES SAVED")
        print("=" * 70)

        print(BINARY_TRAIN_FILE)
        print(BINARY_TEST_FILE)
        print(MULTICLASS_TRAIN_FILE)
        print(MULTICLASS_TEST_FILE)

        print()
        print("=" * 70)
        print("       PHASE 5.3 COMPLETED")
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print("       PHASE 5.3 FAILED")
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()