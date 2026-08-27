import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from google import genai


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RECONCILIATION_DIR = DATA_DIR / "reconciliation"

INPUT_FILE = RECONCILIATION_DIR / "exception_explanations.csv"

TEMPLATE_FILE = (
    RECONCILIATION_DIR / "ai_explanation_templates.csv"
)

OUTPUT_FILE = (
    RECONCILIATION_DIR / "ai_explanations.csv"
)

ENV_FILE = PROJECT_ROOT / ".env"


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.6-flash"

# Your demonstrated free-tier limit is 5 requests/minute.
# 15 seconds between requests keeps us safely below that.
REQUEST_DELAY_SECONDS = 15


# ============================================================
# GEMINI CLIENT
# ============================================================

def create_gemini_client():

    config = dotenv_values(ENV_FILE)

    api_key = config.get("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            f"GEMINI_API_KEY not found in:\n{ENV_FILE}"
        )

    return genai.Client(api_key=api_key)


# ============================================================
# LOAD PHASE 4A DATA
# ============================================================

def load_input_data():

    print("=" * 60)
    print("        LOADING PHASE 4B DATA")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Total records: {len(df)}")

    required_columns = [
        "transaction_id",
        "exception_type",
        "severity",
        "explanation",
        "recommended_action",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("[OK] Input schema valid")

    return df


# ============================================================
# BUILD TEMPLATE PROMPT
# ============================================================

def build_template_prompt(exception_type, severity, sample_rows):

    evidence_text = ""

    for _, row in sample_rows.iterrows():

        evidence_text += (
            f"\nTransaction ID: {row['transaction_id']}"
            f"\nDeterministic explanation: "
            f"{row['explanation']}"
            f"\nRecommended action: "
            f"{row['recommended_action']}"
            f"\n"
        )

    return f"""
You are assisting a financial reconciliation system.

Create a reusable explanation TEMPLATE for this exception type:

Exception type:
{exception_type}

Severity:
{severity}

Below are examples produced by a deterministic
reconciliation engine:
{evidence_text}

IMPORTANT RULES:

1. Do not change the exception type.
2. Do not change its meaning.
3. Do not invent financial facts.
4. Do not invent amounts, dates, IDs, or records.
5. The template will later be filled by Python with
   transaction-specific evidence.
6. Therefore, use these placeholders where appropriate:
   {{transaction_id}}
   {{difference}}
   {{deterministic_explanation}}
   {{recommended_action}}
7. Do not put real transaction IDs or real amounts
   into the template.
8. Keep the wording professional and concise.
9. Return ONLY the reusable template paragraph.
10. Do not use markdown headings or bullet points.

The template should explain what this exception means
and point the analyst toward the recommended action.
"""


# ============================================================
# GENERATE ONE TEMPLATE
# ============================================================

def generate_template(client, exception_type, severity, rows):

    prompt = build_template_prompt(
        exception_type,
        severity,
        rows,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            f"Gemini returned an empty template for "
            f"{exception_type}"
        )

    return response.text.strip()


# ============================================================
# LOAD EXISTING TEMPLATES
# ============================================================

def load_existing_templates():

    if not TEMPLATE_FILE.exists():
        return {}

    try:

        df = pd.read_csv(TEMPLATE_FILE)

        if (
            "exception_type" not in df.columns
            or "severity" not in df.columns
            or "ai_template" not in df.columns
        ):
            return {}

        templates = {}

        for _, row in df.iterrows():

            templates[row["exception_type"]] = {
                "severity": row["severity"],
                "ai_template": row["ai_template"],
            }

        return templates

    except Exception:

        return {}


# ============================================================
# SAVE TEMPLATES
# ============================================================

def save_templates(templates):

    rows = []

    for exception_type, data in templates.items():

        rows.append(
            {
                "exception_type": exception_type,
                "severity": data["severity"],
                "ai_template": data["ai_template"],
            }
        )

    template_df = pd.DataFrame(rows)

    template_df = template_df.sort_values(
        "exception_type"
    )

    template_df.to_csv(
        TEMPLATE_FILE,
        index=False,
    )


# ============================================================
# APPLY TEMPLATE
# ============================================================

def apply_template(row, template):

    text = template

    text = text.replace(
        "{transaction_id}",
        str(row["transaction_id"]),
    )

    text = text.replace(
        "{difference}",
        str(row.get("difference", "")),
    )

    text = text.replace(
        "{deterministic_explanation}",
        str(row["explanation"]),
    )

    text = text.replace(
        "{recommended_action}",
        str(row["recommended_action"]),
    )

    return text.strip()


# ============================================================
# VALIDATE TEMPLATES
# ============================================================

def validate_templates(
    templates,
    exception_types,
):

    missing = [
        exception_type
        for exception_type in exception_types
        if exception_type not in templates
    ]

    if missing:

        raise ValueError(
            "Missing AI templates for: "
            + ", ".join(missing)
        )

    print()
    print("=" * 60)
    print("        TEMPLATE VALIDATION")
    print("=" * 60)

    print(
        f"Required exception types: "
        f"{len(exception_types)}"
    )

    print(
        f"Generated templates: "
        f"{len(templates)}"
    )

    print("[OK] All exception types have templates")


# ============================================================
# VALIDATE FINAL OUTPUT
# ============================================================

def validate_final_output(
    result_df,
    expected_count,
):

    print()
    print("=" * 60)
    print("        FINAL AI OUTPUT VALIDATION")
    print("=" * 60)

    print(
        f"Expected exception records: "
        f"{expected_count}"
    )

    print(
        f"Generated records: "
        f"{len(result_df)}"
    )

    if len(result_df) != expected_count:

        raise ValueError(
            "Final output record count is incorrect."
        )

    if result_df["transaction_id"].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs found."
        )

    if result_df["ai_explanation"].isna().any():

        raise ValueError(
            "Missing AI explanations found."
        )

    if (
        result_df["ai_explanation"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):

        raise ValueError(
            "Empty AI explanations found."
        )

    # --------------------------------------------------------
    # Exception types must remain unchanged
    # --------------------------------------------------------

    original = pd.read_csv(INPUT_FILE)

    original = original[
        original["exception_type"] != "MATCHED"
    ]

    original_map = dict(
        zip(
            original["transaction_id"],
            original["exception_type"],
        )
    )

    for _, row in result_df.iterrows():

        expected_type = original_map.get(
            row["transaction_id"]
        )

        if expected_type != row["exception_type"]:

            raise ValueError(
                f"Exception type changed for "
                f"{row['transaction_id']}"
            )

    print("[OK] Record count valid")
    print("[OK] Transaction IDs unique")
    print("[OK] AI explanations present")
    print("[OK] Exception types preserved")

    print()
    print("Final exception distribution:")

    print(
        result_df["exception_type"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        df = load_input_data()

        exceptions_df = df[
            df["exception_type"] != "MATCHED"
        ].copy()

        print()
        print(
            f"Actual exception records: "
            f"{len(exceptions_df)}"
        )

        if len(exceptions_df) != 200:

            raise ValueError(
                "Expected exactly 200 exception records, "
                f"but found {len(exceptions_df)}."
            )

        # ----------------------------------------------------
        # FIND EXCEPTION TYPES
        # ----------------------------------------------------

        exception_types = sorted(
            exceptions_df[
                "exception_type"
            ].unique()
        )

        print(
            f"Unique exception types: "
            f"{len(exception_types)}"
        )

        print()

        for exception_type in exception_types:
            print(f" - {exception_type}")

        # ----------------------------------------------------
        # LOAD EXISTING TEMPLATES
        # ----------------------------------------------------

        templates = load_existing_templates()

        print()
        print(
            f"Existing templates loaded: "
            f"{len(templates)}"
        )

        # ----------------------------------------------------
        # CREATE GEMINI CLIENT ONLY IF NEEDED
        # ----------------------------------------------------

        missing_types = [
            exception_type
            for exception_type in exception_types
            if exception_type not in templates
        ]

        if missing_types:

            client = create_gemini_client()

            print(
                "\nGemini client initialized successfully."
            )

            print(
                f"Templates still required: "
                f"{len(missing_types)}"
            )

            # ------------------------------------------------
            # GENERATE MISSING TEMPLATES
            # ------------------------------------------------

            for index, exception_type in enumerate(
                missing_types,
                start=1,
            ):

                print()
                print(
                    f"[{index}/{len(missing_types)}] "
                    f"Generating template: "
                    f"{exception_type}"
                )

                type_rows = exceptions_df[
                    exceptions_df[
                        "exception_type"
                    ] == exception_type
                ].head(3)

                severity = type_rows.iloc[0][
                    "severity"
                ]

                template = generate_template(
                    client,
                    exception_type,
                    severity,
                    type_rows,
                )

                templates[exception_type] = {
                    "severity": severity,
                    "ai_template": template,
                }

                print(
                    "[OK] Template generated"
                )

                # Save immediately so successful
                # templates survive interruption.
                save_templates(templates)

                print(
                    f"[OK] Saved progress "
                    f"({len(templates)} templates)"
                )

                if index < len(missing_types):

                    print(
                        f"Waiting "
                        f"{REQUEST_DELAY_SECONDS} "
                        f"seconds for rate-limit protection..."
                    )

                    time.sleep(
                        REQUEST_DELAY_SECONDS
                    )

        else:

            print(
                "\nAll templates already exist."
            )

        # ----------------------------------------------------
        # VALIDATE TEMPLATES
        # ----------------------------------------------------

        validate_templates(
            templates,
            exception_types,
        )

        # ----------------------------------------------------
        # GENERATE 200 FINAL EXPLANATIONS LOCALLY
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("        BUILDING FINAL AI EXPLANATIONS")
        print("=" * 60)

        results = []

        for _, row in exceptions_df.iterrows():

            exception_type = row[
                "exception_type"
            ]

            template = templates[
                exception_type
            ]["ai_template"]

            ai_explanation = apply_template(
                row,
                template,
            )

            results.append(
                {
                    "transaction_id": row[
                        "transaction_id"
                    ],
                    "exception_type": exception_type,
                    "severity": row[
                        "severity"
                    ],
                    "deterministic_explanation": row[
                        "explanation"
                    ],
                    "ai_explanation": ai_explanation,
                    "recommended_action": row[
                        "recommended_action"
                    ],
                }
            )

        result_df = pd.DataFrame(results)

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        validate_final_output(
            result_df,
            expected_count=200,
        )

        # ----------------------------------------------------
        # SAVE FINAL OUTPUT
        # ----------------------------------------------------

        result_df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "AI explanations saved to:"
        )

        print(OUTPUT_FILE)

        # ----------------------------------------------------
        # SAMPLE
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("        SAMPLE AI EXPLANATIONS")
        print("=" * 60)

        print(
            result_df[
                [
                    "transaction_id",
                    "exception_type",
                    "severity",
                    "ai_explanation",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

        print()
        print("=" * 60)
        print("       PHASE 4B COMPLETED")
        print("=" * 60)

        print(
            "Gemini templates generated: "
            f"{len(templates)}"
        )

        print(
            "Final AI explanations: "
            f"{len(result_df)}"
        )

        print(
            "\nGemini was used for exception-type "
            "templates, while transaction-specific "
            "values were preserved by Python."
        )

    except Exception as exc:

        print()
        print("=" * 60)
        print("       PHASE 4B FAILED")
        print("=" * 60)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()