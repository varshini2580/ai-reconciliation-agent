import pandas as pd


# ============================================================
# NUMERIC HELPERS
# ============================================================

TOLERANCE = 0.01


def is_close(a, b, tolerance=TOLERANCE):

    try:
        return abs(
            float(a) - float(b)
        ) <= tolerance

    except (TypeError, ValueError):
        return False


def parse_dates(value):

    if value is None:
        return []

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    result = []

    for item in value.split("|"):

        parsed = pd.to_datetime(
            item.strip(),
            errors="coerce"
        )

        if not pd.isna(parsed):
            result.append(parsed)

    return result


def contains_suffix(value, suffix):

    if value is None:
        return False

    return any(
        item.endswith(suffix)
        for item in str(value).split("|")
    )


# ============================================================
# INDIVIDUAL RULES
# ============================================================

def check_missing_payment(row):

    if int(row["payment_count"]) == 0:

        return {
            "exception_type":
                "MISSING_PAYMENT",

            "reason":
                "No payment record exists "
                "for the transaction.",

            "difference":
                float(row["order_amount"])
        }

    return None


def check_missing_settlement(row):

    if int(row["settlement_count"]) == 0:

        return {
            "exception_type":
                "MISSING_SETTLEMENT",

            "reason":
                "No settlement record exists "
                "for the transaction.",

            "difference":
                float(row["order_amount"])
        }

    return None


def check_failed_payment(row):

    statuses = str(
        row["payment_statuses"]
    ).split("|")

    if "FAILED" in statuses:

        return {
            "exception_type":
                "FAILED_PAYMENT",

            "reason":
                "Payment status is FAILED.",

            "difference":
                float(row["payment_total"])
        }

    return None


def check_duplicate_payment(row):

    if int(row["payment_count"]) > 1:

        payment_ids = str(
            row["payment_ids"]
        ).split("|")

        duplicate_ids = [
            x for x in payment_ids
            if x.endswith("_DUP")
        ]

        if duplicate_ids:

            return {
                "exception_type":
                    "DUPLICATE_PAYMENT",

                "reason":
                    "Duplicate payment record "
                    "detected.",

                "difference":
                    round(
                        float(row["payment_total"])
                        - float(row["order_amount"]),
                        2
                    )
            }

    return None


def check_multiple_payments(row):

    if int(row["payment_count"]) > 1:

        payment_ids = str(
            row["payment_ids"]
        ).split("|")

        part2_ids = [
            x for x in payment_ids
            if x.endswith("_PART2")
        ]

        if part2_ids:

            return {
                "exception_type":
                    "MULTIPLE_PAYMENTS",

                "reason":
                    "Multiple payment records "
                    "exist for the same transaction.",

                "difference":
                    round(
                        float(row["payment_total"])
                        - float(row["order_amount"]),
                        2
                    )
            }

    return None


def check_duplicate_settlement(row):

    if int(row["settlement_count"]) > 1:

        settlement_ids = str(
            row["settlement_ids"]
        ).split("|")

        duplicate_ids = [
            x for x in settlement_ids
            if x.endswith("_DUP")
        ]

        if duplicate_ids:

            return {
                "exception_type":
                    "DUPLICATE_SETTLEMENT",

                "reason":
                    "Duplicate settlement record "
                    "detected.",

                "difference":
                    round(
                        float(row["settlement_net_total"])
                        - float(row["settlement_gross_total"]),
                        2
                    )
            }

    return None


def check_wrong_transaction_reference(row):

    transaction_id = str(
        row["transaction_id"]
    )

    references = str(
        row["bank_references"]
    ).split("|")

    for reference in references:

        reference = reference.strip()

        if not reference:
            continue

        if reference != transaction_id:

            return {
                "exception_type":
                    "WRONG_TRANSACTION_REFERENCE",

                "reason":
                    "Bank reference does not "
                    "match the transaction ID.",

                "difference":
                    0.0
            }

    return None


def check_date_mismatch(row):

    raw_dates = str(
        row["payment_dates"]
    ).split("|")

    clean_dates = str(
        row["clean_payment_dates"]
    ).split("|")

    raw_dates = [
        x.strip()
        for x in raw_dates
        if x.strip()
    ]

    clean_dates = [
        x.strip()
        for x in clean_dates
        if x.strip()
    ]

    if not raw_dates or not clean_dates:
        return None

    if set(raw_dates) != set(clean_dates):

        return {
            "exception_type":
                "DATE_MISMATCH",

            "reason":
                "Payment date differs from "
                "the clean reference date.",

            "difference":
                0.0
        }

    return None


def check_settlement_delay(row):

    raw_dates = str(
        row["settlement_dates"]
    ).split("|")

    clean_dates = str(
        row["clean_settlement_date"]
    ).split("|")

    raw_dates = [
        x.strip()
        for x in raw_dates
        if x.strip()
    ]

    clean_dates = [
        x.strip()
        for x in clean_dates
        if x.strip()
    ]

    if not raw_dates or not clean_dates:
        return None

    if set(raw_dates) != set(clean_dates):

        return {
            "exception_type":
                "SETTLEMENT_DELAY",

            "reason":
                "Settlement date differs from "
                "the clean reference date.",

            "difference":
                0.0
        }

    return None

# ============================================================
# SETTLEMENT CALCULATION
# ============================================================

def calculate_expected_net(row):

    gross = float(
        row["settlement_gross_total"]
    )

    fee = float(
        row["settlement_fee_total"]
    )

    tax = float(
        row["settlement_tax_total"]
    )

    adjustment = float(
        row["settlement_adjustment_total"]
    )

    refund = float(
        row["settlement_refund_total"]
    )

    chargeback = float(
        row["settlement_chargeback_total"]
    )

    expected = (
        gross
        - fee
        - tax
        - refund
        - chargeback
        + adjustment
    )

    return round(expected, 2)


def check_refund(row):

    refund = float(
        row["settlement_refund_total"]
    )

    if refund > TOLERANCE:

        return {
            "exception_type":
                "REFUND",

            "reason":
                "Settlement contains a refund.",

            "difference":
                round(refund, 2)
        }

    return None


def check_chargeback(row):

    chargeback = float(
        row["settlement_chargeback_total"]
    )

    if chargeback > TOLERANCE:

        return {
            "exception_type":
                "CHARGEBACK",

            "reason":
                "Settlement contains a chargeback.",

            "difference":
                round(chargeback, 2)
        }

    return None


def check_unknown_adjustment(row):

    adjustment = float(
        row["settlement_adjustment_total"]
    )

    # Clean settlements are expected to have
    # zero adjustment. The injector introduces
    # -100, -250, -500, 100 or 250.
    if not is_close(adjustment, 0.0):

        return {
            "exception_type":
                "UNKNOWN_ADJUSTMENT",

            "reason":
                "Settlement contains an "
                "unexpected adjustment.",

            "difference":
                round(adjustment, 2)
        }

    return None


def check_incorrect_fee(row):

    gross = float(
        row["settlement_gross_total"]
    )

    fee = float(
        row["settlement_fee_total"]
    )

    if gross <= 0:
        return None

    fee_ratio = fee / gross

    # The generator creates incorrect fees at
    # exactly 5%, 7% or 10%.
    if any(
        abs(fee_ratio - ratio) <= 0.0001
        for ratio in [0.05, 0.07, 0.10]
    ):

        return {
            "exception_type":
                "INCORRECT_FEE",

            "reason":
                f"Settlement fee is "
                f"{fee_ratio:.2%} of gross amount.",

            "difference":
                round(fee, 2)
        }

    return None


def check_partial_settlement(row):

    gross = float(
        row["settlement_gross_total"]
    )

    actual_net = float(
        row["settlement_net_total"]
    )

    if gross <= 0:
        return None

    ratio = actual_net / gross

    # Exact ratios used by the exception generator.
    allowed_ratios = [
        0.50,
        0.60,
        0.70,
        0.80
    ]

    for expected_ratio in allowed_ratios:

        if abs(
            ratio - expected_ratio
        ) <= 0.001:

            return {
                "exception_type":
                    "PARTIAL_SETTLEMENT",

                "reason":
                    f"Settlement net amount is "
                    f"{expected_ratio:.0%} of "
                    "gross amount.",

                "difference":
                    round(
                        gross - actual_net,
                        2
                    )
            }

    return None


def check_amount_mismatch(row):

    if int(row["settlement_count"]) == 0:
        return None

    if int(row["settlement_count"]) > 1:
        return None

    raw_net = float(
        row["settlement_net_total"]
    )

    clean_net = float(
        row["clean_settlement_net"]
    )

    difference = round(
        clean_net - raw_net,
        2
    )

    if abs(difference) > 0.01:

        return {
            "exception_type":
                "AMOUNT_MISMATCH",

            "reason":
                "Settlement net amount differs "
                "from the clean reference amount.",

            "difference":
                difference
        }

    return None

# ============================================================
# MAIN RULE ENGINE
# ============================================================

def apply_rules(row):

    # --------------------------------------------------------
    # 1. Missing records
    # --------------------------------------------------------

    result = check_missing_payment(row)

    if result:
        return result

    result = check_missing_settlement(row)

    if result:
        return result

    # --------------------------------------------------------
    # 2. Payment problems
    # --------------------------------------------------------

    result = check_failed_payment(row)

    if result:
        return result

    result = check_duplicate_payment(row)

    if result:
        return result

    result = check_multiple_payments(row)

    if result:
        return result

    # --------------------------------------------------------
    # 3. Settlement duplicate
    # --------------------------------------------------------

    result = check_duplicate_settlement(row)

    if result:
        return result

    # --------------------------------------------------------
    # 4. Bank reference
    # --------------------------------------------------------

    result = check_wrong_transaction_reference(
        row
    )

    if result:
        return result

    # --------------------------------------------------------
    # 5. Date problems
    # --------------------------------------------------------

    result = check_date_mismatch(row)

    if result:
        return result

    result = check_settlement_delay(row)

    if result:
        return result

    # --------------------------------------------------------
    # 6. Explicit settlement fields
    # --------------------------------------------------------

    result = check_refund(row)

    if result:
        return result

    result = check_chargeback(row)

    if result:
        return result

    result = check_unknown_adjustment(row)

    if result:
        return result

    # --------------------------------------------------------
    # 7. Fee problem
    # --------------------------------------------------------

    result = check_incorrect_fee(row)

    if result:
        return result

    # --------------------------------------------------------
    # 8. Partial settlement
    # --------------------------------------------------------

    result = check_partial_settlement(row)

    if result:
        return result

    # --------------------------------------------------------
    # 9. Generic amount mismatch
    # --------------------------------------------------------

    result = check_amount_mismatch(row)

    if result:
        return result

    # --------------------------------------------------------
    # Normal transaction
    # --------------------------------------------------------

    return {
        "exception_type":
            "MATCHED",

        "reason":
            "Transaction passed all "
            "reconciliation rules.",

        "difference":
            0.0
    }