from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RECON_DIR = DATA_DIR / "reconciliation"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

MATCH_TABLE_FILE = RECON_DIR / "match_table.csv"
GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "ground_truth.csv"
OUTPUT_FILE = RECON_DIR / "ml_dataset.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# Columns that must NEVER become ML input features.
# These contain the answer or directly reveal the answer.
FORBIDDEN_COLUMNS = {
    "transaction_id",
    "exception_type",
    "status",
    "true_status",
    "true_exception_type",
    "true_difference",
    "ground_truth",
}


# ============================================================
# HELPERS
# ============================================================

def require_columns(df, columns, dataset_name):

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )


def safe_numeric(df, column):

    if column not in df.columns:
        return pd.Series(
            0.0,
            index=df.index,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(0.0)


def parse_date(df, column):

    if column not in df.columns:
        return pd.Series(
            pd.NaT,
            index=df.index,
        )

    return pd.to_datetime(
        df[column],
        errors="coerce",
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("        PHASE 5.2 — FEATURE ENGINEERING")
    print("=" * 70)

    if not MATCH_TABLE_FILE.exists():
        raise FileNotFoundError(
            f"Match table not found:\n{MATCH_TABLE_FILE}"
        )

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(
            f"Ground truth not found:\n{GROUND_TRUTH_FILE}"
        )

    match = pd.read_csv(MATCH_TABLE_FILE)
    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

    print(f"Match table records: {len(match)}")
    print(f"Ground truth records: {len(ground_truth)}")

    require_columns(
        match,
        ["transaction_id"],
        "match_table",
    )

    require_columns(
        ground_truth,
        [
            "transaction_id",
            "true_status",
            "true_exception_type",
            "true_difference",
        ],
        "ground_truth",
    )

    print("[OK] Input schemas valid")

    return match, ground_truth


# ============================================================
# MERGE DATA
# ============================================================

def merge_data(match, ground_truth):

    print()
    print("Merging match data with ground truth...")

    if match["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs found in match table."
        )

    if ground_truth["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs found in ground truth."
        )

    df = match.merge(
        ground_truth[
            [
                "transaction_id",
                "true_status",
                "true_exception_type",
                "true_difference",
            ]
        ],
        on="transaction_id",
        how="inner",
        validate="one_to_one",
    )

    print(f"Merged records: {len(df)}")

    if len(df) != len(match):
        raise ValueError(
            "Merge lost records."
        )

    print("[OK] One-to-one merge successful")

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    print()
    print("Creating leakage-safe ML features...")

    features = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # BASIC AMOUNTS
    # --------------------------------------------------------

    order_amount = safe_numeric(
        df,
        "order_amount",
    )

    payment_total = safe_numeric(
        df,
        "payment_total",
    )

    bank_total = safe_numeric(
        df,
        "bank_total",
    )

    settlement_gross = safe_numeric(
        df,
        "settlement_gross_total",
    )

    settlement_net = safe_numeric(
        df,
        "settlement_net_total",
    )

    settlement_fee = safe_numeric(
        df,
        "settlement_fee_total",
    )

    settlement_tax = safe_numeric(
        df,
        "settlement_tax_total",
    )

    settlement_adjustment = safe_numeric(
        df,
        "settlement_adjustment_total",
    )

    settlement_refund = safe_numeric(
        df,
        "settlement_refund_total",
    )

    settlement_chargeback = safe_numeric(
        df,
        "settlement_chargeback_total",
    )

    # --------------------------------------------------------
    # AMOUNT DIFFERENCES
    # --------------------------------------------------------

    features["order_amount"] = order_amount

    features["payment_total"] = payment_total

    features["bank_total"] = bank_total

    features["settlement_gross_total"] = (
        settlement_gross
    )

    features["settlement_net_total"] = (
        settlement_net
    )

    features["settlement_fee_total"] = (
        settlement_fee
    )

    features["settlement_tax_total"] = (
        settlement_tax
    )

    features["settlement_adjustment_total"] = (
        settlement_adjustment
    )

    features["settlement_refund_total"] = (
        settlement_refund
    )

    features["settlement_chargeback_total"] = (
        settlement_chargeback
    )

    features["payment_order_difference"] = (
        payment_total - order_amount
    )

    features["bank_order_difference"] = (
        bank_total - order_amount
    )

    features["settlement_gross_order_difference"] = (
        settlement_gross - order_amount
    )

    features["settlement_net_order_difference"] = (
        settlement_net - order_amount
    )

    features["payment_bank_difference"] = (
        payment_total - bank_total
    )

    features["bank_settlement_difference"] = (
        bank_total - settlement_gross
    )

    # --------------------------------------------------------
    # RATIOS
    # --------------------------------------------------------

    features["payment_order_ratio"] = np.where(
        order_amount != 0,
        payment_total / order_amount,
        0.0,
    )

    features["bank_order_ratio"] = np.where(
        order_amount != 0,
        bank_total / order_amount,
        0.0,
    )

    features["settlement_gross_order_ratio"] = np.where(
        order_amount != 0,
        settlement_gross / order_amount,
        0.0,
    )

    features["settlement_net_order_ratio"] = np.where(
        order_amount != 0,
        settlement_net / order_amount,
        0.0,
    )

    # --------------------------------------------------------
    # RECORD COUNTS
    # --------------------------------------------------------

    features["payment_count"] = safe_numeric(
        df,
        "payment_count",
    )

    features["bank_transaction_count"] = safe_numeric(
        df,
        "bank_transaction_count",
    )

    features["settlement_count"] = safe_numeric(
        df,
        "settlement_count",
    )

    # --------------------------------------------------------
    # BOOLEAN / STRUCTURAL FEATURES
    # --------------------------------------------------------

    features["has_payment"] = (
        features["payment_count"] > 0
    ).astype(int)

    features["has_settlement"] = (
        features["settlement_count"] > 0
    ).astype(int)

    features["has_multiple_payments"] = (
        features["payment_count"] > 1
    ).astype(int)

    features["has_multiple_settlements"] = (
        features["settlement_count"] > 1
    ).astype(int)

    features["has_multiple_bank_transactions"] = (
        features["bank_transaction_count"] > 1
    ).astype(int)

    features["has_refund"] = (
        settlement_refund != 0
    ).astype(int)

    features["has_chargeback"] = (
        settlement_chargeback != 0
    ).astype(int)

    features["has_adjustment"] = (
        settlement_adjustment != 0
    ).astype(int)

    features["has_fee"] = (
        settlement_fee != 0
    ).astype(int)

    features["has_tax"] = (
        settlement_tax != 0
    ).astype(int)

    # --------------------------------------------------------
    # PAYMENT STATUS
    # --------------------------------------------------------

    if "payment_statuses" in df.columns:

        status = (
            df["payment_statuses"]
            .fillna("")
            .astype(str)
            .str.upper()
        )

        features["payment_success"] = (
            status.str.contains(
                "SUCCESS",
                regex=False,
            )
        ).astype(int)

        features["payment_failed"] = (
            status.str.contains(
                "FAILED",
                regex=False,
            )
        ).astype(int)

    else:

        features["payment_success"] = 0
        features["payment_failed"] = 0

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    order_date = parse_date(
        df,
        "order_date",
    )

    payment_date = parse_date(
        df,
        "clean_payment_dates",
    )

    settlement_date = parse_date(
        df,
        "clean_settlement_date",
    )

    features["payment_order_delay_days"] = (
        payment_date - order_date
    ).dt.days.fillna(0)

    features["settlement_order_delay_days"] = (
        settlement_date - order_date
    ).dt.days.fillna(0)

    features["settlement_payment_delay_days"] = (
        settlement_date - payment_date
    ).dt.days.fillna(0)

    features["payment_date_available"] = (
        payment_date.notna()
    ).astype(int)

    features["settlement_date_available"] = (
        settlement_date.notna()
    ).astype(int)

    # --------------------------------------------------------
    # DATE ANOMALY FLAGS
    # --------------------------------------------------------

    features["payment_before_order"] = (
        features["payment_order_delay_days"] < 0
    ).astype(int)

    features["settlement_before_order"] = (
        features["settlement_order_delay_days"] < 0
    ).astype(int)

    features["settlement_before_payment"] = (
        features["settlement_payment_delay_days"] < 0
    ).astype(int)

    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    if "payment_method" in df.columns:

        features["payment_method"] = (
            df["payment_method"]
            .fillna("UNKNOWN")
            .astype(str)
        )

    elif "payment_methods" in df.columns:

        features["payment_method"] = (
            df["payment_methods"]
            .fillna("UNKNOWN")
            .astype(str)
        )

    else:

        features["payment_method"] = "UNKNOWN"

    if "currency" in df.columns:

        features["currency"] = (
            df["currency"]
            .fillna("UNKNOWN")
            .astype(str)
        )

    else:

        features["currency"] = "UNKNOWN"

    # --------------------------------------------------------
    # MERCHANT INFORMATION
    # --------------------------------------------------------

    if "merchant_id" in df.columns:

        features["merchant_id"] = (
            df["merchant_id"]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    forbidden_present = (
        FORBIDDEN_COLUMNS
        .intersection(features.columns)
    )

    if forbidden_present:

        raise ValueError(
            "LEAKAGE DETECTED. Forbidden columns found "
            f"in features: {forbidden_present}"
        )

    print(
        f"[OK] Created {features.shape[1]} features"
    )

    return features


# ============================================================
# BUILD TARGETS
# ============================================================

def create_targets(df):

    print()
    print("Creating ML targets...")

    # --------------------------------------------------------
    # BINARY TARGET
    # --------------------------------------------------------

    target_status = (
        df["true_status"]
        .astype(str)
        .str.upper()
    )

    binary_target = (
        target_status == "EXCEPTION"
    ).astype(int)

    # --------------------------------------------------------
    # MULTICLASS TARGET
    # --------------------------------------------------------

    multiclass_target = (
        df["true_exception_type"]
        .astype(str)
        .str.upper()
    )

    print(
        "Binary target distribution:"
    )

    print(
        binary_target
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Multiclass target distribution:"
    )

    print(
        multiclass_target
        .value_counts()
        .sort_index()
    )

    return binary_target, multiclass_target


# ============================================================
# FINAL DATASET
# ============================================================

def build_dataset(df, features, binary_target, multiclass_target):

    result = features.copy()

    # Keep transaction ID only as an identifier.
    result.insert(
        0,
        "transaction_id",
        df["transaction_id"].values,
    )

    result["target_exception"] = (
        binary_target.values
    )

    result["target_exception_type"] = (
        multiclass_target.values
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(
    ml_dataset,
    source_df,
):

    print()
    print("=" * 70)
    print("        PHASE 5.2 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    print(
        f"Source records: {len(source_df)}"
    )

    print(
        f"ML records: {len(ml_dataset)}"
    )

    if len(ml_dataset) != len(source_df):

        raise ValueError(
            "ML dataset record count mismatch."
        )

    print("[OK] Record count valid")

    # --------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------

    duplicate_ids = (
        ml_dataset["transaction_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids:

        raise ValueError(
            f"Duplicate transaction IDs: "
            f"{duplicate_ids}"
        )

    print("[OK] Transaction IDs unique")

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    valid_binary = {
        0,
        1,
    }

    actual_binary = set(
        ml_dataset["target_exception"]
    )

    if not actual_binary.issubset(valid_binary):

        raise ValueError(
            f"Invalid binary targets: "
            f"{actual_binary}"
        )

    print("[OK] Binary target valid")

    # --------------------------------------------------------
    # Exception target
    # --------------------------------------------------------

    exception_rows = ml_dataset[
        ml_dataset["target_exception"] == 1
    ]

    non_exception_rows = ml_dataset[
        ml_dataset["target_exception"] == 0
    ]

    print()
    print(
        f"Exception records: "
        f"{len(exception_rows)}"
    )

    print(
        f"Matched records: "
        f"{len(non_exception_rows)}"
    )

    if len(exception_rows) != 200:
        raise ValueError(
            "Expected exactly 200 exception records."
        )

    if len(non_exception_rows) != 800:
        raise ValueError(
            "Expected exactly 800 matched records."
        )

    print("[OK] Target counts match ground truth")

    # --------------------------------------------------------
    # Null check
    # --------------------------------------------------------

    null_counts = (
        ml_dataset
        .drop(columns=["transaction_id"])
        .isna()
        .sum()
    )

    problematic = null_counts[
        null_counts > 0
    ]

    if len(problematic):

        print(
            "Warning: missing values found:"
        )

        print(problematic.to_string())

    else:

        print("[OK] No missing feature values")

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    leakage = (
        FORBIDDEN_COLUMNS
        .intersection(
            ml_dataset.columns
        )
    )

    # transaction_id is allowed as identifier,
    # but not as an actual feature.
    leakage.discard("transaction_id")

    if leakage:

        raise ValueError(
            f"Target leakage detected: {leakage}"
        )

    print("[OK] Leakage check passed")

    # --------------------------------------------------------
    # Target type distribution
    # --------------------------------------------------------

    print()
    print(
        "Exception type distribution:"
    )

    print(
        ml_dataset[
            ml_dataset["target_exception"] == 1
        ]["target_exception_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in ml_dataset.columns
        if column not in {
            "transaction_id",
            "target_exception",
            "target_exception_type",
        }
    ]

    print()
    print(
        f"ML feature count: "
        f"{len(feature_columns)}"
    )

    print(
        "[OK] Phase 5.2 feature dataset validated"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        match, ground_truth = load_data()

        # ----------------------------------------------------
        # MERGE
        # ----------------------------------------------------

        merged = merge_data(
            match,
            ground_truth,
        )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        features = create_features(
            merged
        )

        # ----------------------------------------------------
        # TARGETS
        # ----------------------------------------------------

        binary_target, multiclass_target = (
            create_targets(merged)
        )

        # ----------------------------------------------------
        # FINAL DATASET
        # ----------------------------------------------------

        ml_dataset = build_dataset(
            merged,
            features,
            binary_target,
            multiclass_target,
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        validate_dataset(
            ml_dataset,
            merged,
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        ml_dataset.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "ML dataset saved to:"
        )

        print(OUTPUT_FILE)

        # ----------------------------------------------------
        # SAMPLE
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("        ML DATASET SAMPLE")
        print("=" * 70)

        print(
            ml_dataset
            .head(5)
            .to_string(index=False)
        )

        print()
        print("=" * 70)
        print("       PHASE 5.2 COMPLETED")
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print("       PHASE 5.2 FAILED")
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()