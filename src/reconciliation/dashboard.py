from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RECON_DIR = PROJECT_ROOT / "data" / "reconciliation"


DASHBOARD_FILE = RECON_DIR / "dashboard_data.csv"
KPI_FILE = RECON_DIR / "dashboard_kpis.csv"
EXCEPTION_FILE = RECON_DIR / "exception_analytics.csv"
ML_AGENT_FILE = RECON_DIR / "ml_agent_analytics.csv"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Reconciliation Agent",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    dashboard = pd.read_csv(
        DASHBOARD_FILE
    )

    kpis = pd.read_csv(
        KPI_FILE
    )

    exceptions = pd.read_csv(
        EXCEPTION_FILE
    )

    ml_agent = pd.read_csv(
        ML_AGENT_FILE
    )

    return (
        dashboard,
        kpis,
        exceptions,
        ml_agent,
    )


# ============================================================
# KPI HELPER
# ============================================================

def get_kpi(kpis, name):

    rows = kpis[
        kpis["kpi_name"] == name
    ]

    if rows.empty:
        return 0

    return rows.iloc[0]["kpi_value"]


# ============================================================
# LOAD
# ============================================================

try:

    (
        dashboard,
        kpis,
        exceptions,
        ml_agent,
    ) = load_data()

except Exception as exc:

    st.error(
        f"Unable to load dashboard data: {exc}"
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🔍 AI-Powered Reconciliation Dashboard"
)

st.caption(
    "Hybrid Rule + ML + Agent-based Reconciliation System"
)


st.divider()


# ============================================================
# TOP KPIs
# ============================================================

total_transactions = get_kpi(
    kpis,
    "TOTAL_TRANSACTIONS",
)

matched_transactions = get_kpi(
    kpis,
    "MATCHED_TRANSACTIONS",
)

exception_transactions = get_kpi(
    kpis,
    "EXCEPTION_TRANSACTIONS",
)

reconciliation_rate = get_kpi(
    kpis,
    "RECONCILIATION_RATE_PERCENT",
)

escalated_cases = get_kpi(
    kpis,
    "ESCALATED_CASES",
)

pending_reviews = get_kpi(
    kpis,
    "PENDING_REVIEW_CASES",
)

ml_disagreements = get_kpi(
    kpis,
    "ML_DISAGREEMENTS",
)

st.subheader(
    "Reconciliation Overview"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{int(total_transactions):,}",
)

col2.metric(
    "Matched",
    f"{int(matched_transactions):,}",
)

col3.metric(
    "Exceptions",
    f"{int(exception_transactions):,}",
)

col4.metric(
    "Reconciliation Rate",
    f"{reconciliation_rate:.1f}%",
)


col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Escalated Cases",
    f"{int(escalated_cases):,}",
)

col6.metric(
    "Pending Reviews",
    f"{int(pending_reviews):,}",
)

col7.metric(
    "ML Disagreements",
    f"{int(ml_disagreements):,}",
)

col8.metric(
    "Exception Rate",
    f"{get_kpi(kpis, 'EXCEPTION_RATE_PERCENT'):.1f}%",
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header(
    "Dashboard Filters"
)


status_options = sorted(
    dashboard["status"]
    .dropna()
    .unique()
    .tolist()
)

selected_status = st.sidebar.multiselect(
    "Status",
    options=status_options,
    default=status_options,
)


severity_options = sorted(
    dashboard["severity"]
    .dropna()
    .unique()
    .tolist()
)

selected_severity = st.sidebar.multiselect(
    "Severity",
    options=severity_options,
    default=severity_options,
)


exception_options = sorted(
    dashboard["exception_type"]
    .dropna()
    .unique()
    .tolist()
)

selected_exceptions = st.sidebar.multiselect(
    "Exception Type",
    options=exception_options,
    default=exception_options,
)


# ============================================================
# FILTER DATA
# ============================================================

filtered = dashboard[
    dashboard["status"].isin(
        selected_status
    )
].copy()


if selected_severity:

    filtered = filtered[
        (
            filtered["severity"].isna()
        )
        |
        (
            filtered["severity"].isin(
                selected_severity
            )
        )
    ]


if selected_exceptions:

    filtered = filtered[
        (
            filtered["exception_type"].isna()
        )
        |
        (
            filtered["exception_type"].isin(
                selected_exceptions
            )
        )
    ]


# ============================================================
# EXCEPTION ANALYTICS
# ============================================================

st.divider()

st.subheader(
    "Exception Analytics"
)


exception_view = (
    filtered[
        filtered["is_exception"] == 1
    ]
    .groupby(
        "exception_type"
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        "count",
        ascending=False,
    )
)


col_left, col_right = st.columns(2)


with col_left:

    st.write(
        "Exceptions by Type"
    )

    if not exception_view.empty:

        chart_data = (
            exception_view
            .set_index(
                "exception_type"
            )
        )

        st.bar_chart(
            chart_data["count"]
        )

    else:

        st.info(
            "No exception records match the filters."
        )


with col_right:

    st.write(
        "Exceptions by Severity"
    )

    severity_view = (
        filtered[
            filtered["is_exception"] == 1
        ]
        .groupby(
            "severity"
        )
        .size()
        .reset_index(
            name="count"
        )
        .set_index(
            "severity"
        )
    )

    if not severity_view.empty:

        st.bar_chart(
            severity_view["count"]
        )

    else:

        st.info(
            "No severity data available."
        )


# ============================================================
# HYBRID DECISION ANALYTICS
# ============================================================

st.divider()

st.subheader(
    "Hybrid Rule + ML Analysis"
)


hybrid_view = (
    filtered
    .groupby(
        "hybrid_decision"
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        "count",
        ascending=False,
    )
)


if not hybrid_view.empty:

    st.bar_chart(
        hybrid_view.set_index(
            "hybrid_decision"
        )["count"]
    )


# ============================================================
# AGENT ANALYTICS
# ============================================================

st.divider()

st.subheader(
    "Agent Workflow"
)


agent_view = (
    filtered[
        filtered["is_exception"] == 1
    ]
    .groupby(
        "agent_decision"
    )
    .size()
    .reset_index(
        name="count"
    )
)


col_left, col_right = st.columns(2)


with col_left:

    st.write(
        "Agent Decisions"
    )

    if not agent_view.empty:

        st.bar_chart(
            agent_view.set_index(
                "agent_decision"
            )["count"]
        )

    else:

        st.info(
            "No agent decisions match the filters."
        )


with col_right:

    st.write(
        "Escalation Priority"
    )

    escalation_view = (
        filtered[
            filtered["is_escalated"] == 1
        ]
        .groupby(
            "escalation_priority"
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    if not escalation_view.empty:

        st.bar_chart(
            escalation_view.set_index(
                "escalation_priority"
            )["count"]
        )

    else:

        st.info(
            "No escalations match the filters."
        )


# ============================================================
# ML SIGNALS
# ============================================================

st.divider()

st.subheader(
    "ML Signals"
)


ml_view = (
    filtered[
        filtered["is_exception"] == 1
    ]
    .groupby(
        "hybrid_decision"
    )
    .agg(
        cases=(
            "transaction_id",
            "count",
        ),
        ml_support=(
            "ml_supported_exception",
            "sum",
        ),
        ml_disagreement=(
            "ml_disagreement",
            "sum",
        ),
    )
    .reset_index()
)


st.dataframe(
    ml_view,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FINANCIAL IMPACT
# ============================================================

st.divider()

st.subheader(
    "Financial Impact"
)


financial_df = filtered[
    filtered["is_exception"] == 1
].copy()


if not financial_df.empty:

    financial_df[
        "difference"
    ] = pd.to_numeric(
        financial_df["difference"],
        errors="coerce",
    )

    total_difference = (
        financial_df["difference"]
        .sum()
    )

    absolute_difference = (
        financial_df["difference"]
        .abs()
        .sum()
    )

    avg_difference = (
        financial_df["difference"]
        .mean()
    )

else:

    total_difference = 0
    absolute_difference = 0
    avg_difference = 0


col1, col2, col3 = st.columns(3)


col1.metric(
    "Total Difference",
    f"{total_difference:,.2f}",
)

col2.metric(
    "Absolute Difference",
    f"{absolute_difference:,.2f}",
)

col3.metric(
    "Average Difference",
    f"{avg_difference:,.2f}",
)


# ============================================================
# EXCEPTION TABLE
# ============================================================

st.divider()

st.subheader(
    "Exception Cases"
)


display_columns = [
    "transaction_id",
    "exception_type",
    "severity",
    "difference",
    "hybrid_decision",
    "agent_decision",
    "escalation_required",
    "escalation_priority",
    "review_status",
]


available_columns = [
    column
    for column in display_columns
    if column in filtered.columns
]


exception_table = filtered[
    filtered["is_exception"] == 1
][available_columns]


st.dataframe(
    exception_table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Execution mode: SIMULATED | "
    "Deterministic rules remain authoritative | "
    "ML provides supporting signals | "
    "Human review required for agent actions"
)