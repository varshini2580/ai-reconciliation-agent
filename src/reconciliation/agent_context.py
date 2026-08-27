from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"

RECONCILIATION_FILE = (
    RECON_DIR / "reconciliation_results.csv"
)

EXPLANATION_FILE = (
    RECON_DIR / "exception_explanations.csv"
)

RESOLUTION_FILE = (
    RECON_DIR / "resolution_actions.csv"
)

HYBRID_FILE = (
    RECON_DIR / "hybrid_decisions.csv"
)

OUTPUT_FILE = (
    RECON_DIR / "agent_context.csv"
)


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():

    print("=" * 70)
    print("        PHASE 6.1 — AGENT CONTEXT BUILDER")
    print("=" * 70)

    files = {
        "reconciliation": RECONCILIATION_FILE,
        "explanation": EXPLANATION_FILE,
        "resolution": RESOLUTION_FILE,
        "hybrid": HYBRID_FILE,
    }

    for name, path in files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

    reconciliation = pd.read_csv(
        RECONCILIATION_FILE
    )

    explanation = pd.read_csv(
        EXPLANATION_FILE
    )

    resolution = pd.read_csv(
        RESOLUTION_FILE
    )

    hybrid = pd.read_csv(
        HYBRID_FILE
    )

    print(
        f"Reconciliation records: {len(reconciliation)}"
    )

    print(
        f"Explanation records: {len(explanation)}"
    )

    print(
        f"Resolution records: {len(resolution)}"
    )

    print(
        f"Hybrid records: {len(hybrid)}"
    )

    print("[OK] All Phase 6 inputs loaded")

    return (
        reconciliation,
        explanation,
        resolution,
        hybrid,
    )


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    reconciliation,
    explanation,
    resolution,
    hybrid,
):

    print()
    print("========== INPUT VALIDATION ==========")

    datasets = {
        "reconciliation": reconciliation,
        "explanation": explanation,
        "resolution": resolution,
        "hybrid": hybrid,
    }

    for name, df in datasets.items():

        if "transaction_id" not in df.columns:

            raise ValueError(
                f"{name} missing transaction_id"
            )

        if df[
            "transaction_id"
        ].duplicated().any():

            raise ValueError(
                f"{name} contains duplicate "
                "transaction IDs"
            )

        print(
            f"[OK] {name} schema valid"
        )


# ============================================================
# BUILD AGENT CONTEXT
# ============================================================

def build_context(
    reconciliation,
    explanation,
    resolution,
    hybrid,
):

    print()
    print("========== BUILDING AGENT CONTEXT ==========")

    # --------------------------------------------------------
    # Select only the fields required by the agent.
    # --------------------------------------------------------

    reconciliation_fields = [
        "transaction_id",
        "status",
        "exception_type",
        "difference",
    ]

    explanation_fields = [
        "transaction_id",
        "severity",
        "explanation",
        "recommended_action",
    ]

    resolution_fields = [
        "transaction_id",
        "priority",
        "resolution_category",
        "next_step",
        "escalation_required",
    ]

    hybrid_fields = [
        "transaction_id",
        "ml_status",
        "ml_exception_probability",
        "ml_predicted_exception_type",
        "ml_exception_type_probability",
        "ml_agrees_with_rule_status",
        "ml_agrees_with_rule_exception_type",
        "hybrid_decision",
    ]

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    for name, df, fields in [
        (
            "reconciliation",
            reconciliation,
            reconciliation_fields,
        ),
        (
            "explanation",
            explanation,
            explanation_fields,
        ),
        (
            "resolution",
            resolution,
            resolution_fields,
        ),
        (
            "hybrid",
            hybrid,
            hybrid_fields,
        ),
    ]:

        missing = [
            field
            for field in fields
            if field not in df.columns
        ]

        if missing:

            raise ValueError(
                f"{name} missing required columns: "
                f"{missing}"
            )

    # --------------------------------------------------------
    # Filter to actual exceptions.
    # --------------------------------------------------------

    exceptions = reconciliation[
        reconciliation["status"]
        == "EXCEPTION"
    ][
        reconciliation_fields
    ].copy()

    print(
        f"Exception cases: {len(exceptions)}"
    )

    # --------------------------------------------------------
    # Merge explanation.
    # --------------------------------------------------------

    context = exceptions.merge(
        explanation[
            explanation_fields
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Merge resolution.
    # --------------------------------------------------------

    context = context.merge(
        resolution[
            resolution_fields
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Merge hybrid decision.
    # --------------------------------------------------------

    context = context.merge(
        hybrid[
            hybrid_fields
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Agent context records: {len(context)}"
    )

    return context


# ============================================================
# VALIDATE CONTEXT
# ============================================================

def validate_context(context):

    print()
    print("=" * 70)
    print("        PHASE 6.1 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    if len(context) != 200:

        raise ValueError(
            f"Expected 200 exception cases, "
            f"found {len(context)}"
        )

    print(
        "[OK] 200 exception cases present"
    )

    # --------------------------------------------------------
    # Unique transactions
    # --------------------------------------------------------

    if context[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs found"
        )

    print(
        "[OK] Transaction IDs unique"
    )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required = [
        "transaction_id",
        "status",
        "exception_type",
        "difference",
        "severity",
        "explanation",
        "recommended_action",
        "priority",
        "resolution_category",
        "next_step",
        "escalation_required",
        "ml_status",
        "ml_exception_probability",
        "ml_predicted_exception_type",
        "ml_exception_type_probability",
        "hybrid_decision",
    ]

    missing = [
        column
        for column in required
        if column not in context.columns
    ]

    if missing:

        raise ValueError(
            f"Missing context fields: {missing}"
        )

    print(
        "[OK] Required agent fields present"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    important_fields = [
        "exception_type",
        "severity",
        "explanation",
        "recommended_action",
        "priority",
        "resolution_category",
        "next_step",
        "escalation_required",
        "hybrid_decision",
    ]

    missing_values = (
        context[important_fields]
        .isna()
        .sum()
        .sum()
    )

    if missing_values:

        raise ValueError(
            f"Missing values detected: "
            f"{missing_values}"
        )

    print(
        "[OK] No missing critical context values"
    )

    # --------------------------------------------------------
    # Exception distribution
    # --------------------------------------------------------

    print()
    print(
        "Exception distribution:"
    )

    print(
        context[
            "exception_type"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Severity distribution
    # --------------------------------------------------------

    print()
    print(
        "Severity distribution:"
    )

    print(
        context[
            "severity"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Hybrid decision distribution
    # --------------------------------------------------------

    print()
    print(
        "Hybrid decision distribution:"
    )

    print(
        context[
            "hybrid_decision"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        (
            reconciliation,
            explanation,
            resolution,
            hybrid,
        ) = load_inputs()

        validate_inputs(
            reconciliation,
            explanation,
            resolution,
            hybrid,
        )

        context = build_context(
            reconciliation,
            explanation,
            resolution,
            hybrid,
        )

        validate_context(
            context
        )

        context.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Agent context saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print("=" * 70)
        print(
            "       PHASE 6.1 COMPLETED"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "       PHASE 6.1 FAILED"
        )
        print("=" * 70)

        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()