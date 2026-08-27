from pathlib import Path
import os
import time

import pandas as pd
from openai import OpenAI


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RECONCILIATION_DIR = DATA_DIR / "reconciliation"

INPUT_FILE = (
    RECONCILIATION_DIR
    / "exception_explanations.csv"
)

OUTPUT_FILE = (
    RECONCILIATION_DIR
    / "ai_explanations.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gpt-5.6-luna"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Exception explanations not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "transaction_id",
        "status",
        "exception_type",
        "severity",
        "explanation",
        "evidence",
        "difference",
        "recommended_action",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + ", ".join(missing)
        )

    return df


# ============================================================
# CREATE AI PROMPT
# ============================================================

def create_prompt(row):

    return f"""
You are an AI assistant for a financial transaction
reconciliation system.

Rewrite the following deterministic reconciliation result
into a short, professional explanation for a finance user.

IMPORTANT RULES:

1. Do NOT change the exception type.
2. Do NOT change any numerical values.
3. Do NOT invent missing information.
4. Do NOT create new financial conclusions.
5. Use only the information provided below.
6. Keep the explanation concise and easy to understand.
7. Mention what happened and why it matters.
8. Include the recommended action.
9. If the status is MATCHED, simply explain that the
   transaction passed reconciliation checks.

Transaction ID:
{row["transaction_id"]}

Status:
{row["status"]}

Exception type:
{row["exception_type"]}

Severity:
{row["severity"]}

Deterministic explanation:
{row["explanation"]}

Evidence:
{row["evidence"]}

Difference:
{row["difference"]}

Recommended action:
{row["recommended_action"]}

Return ONLY the final explanation in plain text.
Do not use JSON.
Do not add headings.
"""


# ============================================================
# GENERATE AI EXPLANATION
# ============================================================

def generate_ai_explanation(
    client,
    row
):

    prompt = create_prompt(row)

    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    text = response.output_text.strip()

    if not text:

        raise ValueError(
            f"Empty AI response for "
            f"{row['transaction_id']}"
        )

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "================================================"
    )

    print(
        "       AI EXPLANATION GENERATION"
    )

    print(
        "================================================"
    )

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    if not os.getenv("OPENAI_API_KEY"):

        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )

    client = OpenAI()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    print(
        f"Input explanation records: {len(df)}"
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    ai_explanations = []

    total = len(df)

    for index, row in df.iterrows():

        transaction_id = row["transaction_id"]

        print(
            f"[{index + 1}/{total}] "
            f"Generating explanation for "
            f"{transaction_id}..."
        )

        try:

            ai_text = generate_ai_explanation(
                client,
                row
            )

            ai_explanations.append(
                ai_text
            )

        except Exception as error:

            print(
                f"ERROR for {transaction_id}: "
                f"{error}"
            )

            # Preserve deterministic explanation
            # instead of losing the transaction.
            ai_explanations.append(
                row["explanation"]
            )

        # Small delay to avoid sending requests
        # too aggressively.
        time.sleep(0.1)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = df.copy()

    output[
        "ai_explanation"
    ] = ai_explanations

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\n========== AI EXPLANATION VALIDATION =========="
    )

    print(
        f"Input records: "
        f"{len(df)}"
    )

    print(
        f"Output records: "
        f"{len(output)}"
    )

    print(
        f"Unique transactions: "
        f"{output['transaction_id'].nunique()}"
    )

    empty_count = (
        output["ai_explanation"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"Empty AI explanations: "
        f"{empty_count}"
    )

    if len(output) != len(df):

        raise ValueError(
            "Output record count does not match input."
        )

    if (
        output["transaction_id"].nunique()
        != len(output)
    ):

        raise ValueError(
            "Duplicate transaction IDs found."
        )

    if empty_count > 0:

        raise ValueError(
            "Empty AI explanations found."
        )

    print(
        "\nAI explanation generation successful."
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    print(
        "\n========== SAMPLE AI EXPLANATIONS =========="
    )

    sample = output[
        output["status"] == "EXCEPTION"
    ].head(5)

    for _, row in sample.iterrows():

        print(
            f"\nTransaction: "
            f"{row['transaction_id']}"
        )

        print(
            f"Exception: "
            f"{row['exception_type']}"
        )

        print(
            f"AI Explanation: "
            f"{row['ai_explanation']}"
        )

    print(
        "\n================================================"
    )

    print(
        "       PHASE 4B COMPLETED"
    )

    print(
        "================================================"
    )


if __name__ == "__main__":
    main()