import streamlit as st
import pandas as pd
from pathlib import Path
from textwrap import dedent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RazorRecon AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "reconciliation"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 115px;
    }

    .status-label {
        font-size: 14px;
        opacity: 0.7;
    }

    .status-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
    }

    .flow-box {
        padding: 16px 10px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 90px;
    }

    .flow-title {
        font-weight: 700;
        font-size: 16px;
    }

    .flow-subtitle {
        font-size: 12px;
        opacity: 0.7;
        margin-top: 5px;
    }

    .safe-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(50,180,100,0.4);
        background: rgba(50,180,100,0.08);
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(240,170,50,0.4);
        background: rgba(240,170,50,0.08);
    }

    .table-wrapper {
        overflow-x: auto;
    }

    .dashboard-table {
        width: 100%;
        border-collapse: collapse;
    }

    .dashboard-table th,
    .dashboard-table td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid rgba(128,128,128,0.25);
        text-align: left;
    }

    .dashboard-table th {
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


dashboard = load_csv("dashboard_data.csv")
kpis = load_csv("dashboard_kpis.csv")
exception_analytics = load_csv("exception_analytics.csv")
ml_agent = load_csv("ml_agent_analytics.csv")
reconciliation_results = load_csv("reconciliation_results.csv")
hybrid = load_csv("hybrid_decisions.csv")
agent = load_csv("agent_decisions.csv")
actions = load_csv("agent_actions.csv")
escalations = load_csv("escalation_cases.csv")

audit = load_csv("audit_events.csv")
review = load_csv("review_queue.csv")
explanations = load_csv("exception_explanations.csv")
ai_explanations = load_csv("ai_explanations.csv")


if dashboard.empty:
    st.error(
        "dashboard_data.csv could not be loaded. "
        "Please run the Phase 7 dashboard pipeline first."
    )
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def number(value):
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return "0"


def money(value):
    try:
        return f"₹{float(value):,.2f}"
    except Exception:
        return "₹0.00"


def get_counts(df, column):
    if column and column in df.columns:
        return df[column].dropna().value_counts()
    return pd.Series(dtype="int64")


def render_dataframe(dataframe):
    html = dataframe.to_html(index=False, escape=True, classes="dashboard-table")
    st.markdown(
        f'<div class="table-wrapper">{html}</div>',
        unsafe_allow_html=True,
    )


def render_count_table(title, counts):
    """Render lightweight native Streamlit bar visualization."""
    if counts.empty:
        st.info(f"No {title.lower()} data available.")
        return

    st.subheader(title.title())

    counts = counts.copy().sort_values(ascending=True)
    max_count = max(float(counts.max()), 1.0)

    for category, count in counts.items():
        value = float(count)
        st.markdown(f"**{category}** — {int(value):,}")
        st.progress(min(value / max_count, 1.0))


# DASHBOARD METRICS
# ============================================================

status_col = find_column(dashboard, ["status"])
exception_col = find_column(dashboard, ["exception_type"])
severity_col = find_column(dashboard, ["severity"])
hybrid_col = find_column(dashboard, ["hybrid_decision"])
agent_col = find_column(dashboard, ["agent_decision"])

total = len(dashboard)

matched = (
    int((dashboard[status_col] == "MATCHED").sum())
    if status_col
    else 0
)

exceptions = (
    int((dashboard[status_col] == "EXCEPTION").sum())
    if status_col
    else 0
)

escalated = (
    int((dashboard[agent_col] == "ESCALATE_FOR_REVIEW").sum())
    if agent_col
    else len(escalations)
)

disagreements = (
    int(
        (
            dashboard[hybrid_col]
            == "RULE_EXCEPTION_ML_DISAGREEMENT"
        ).sum()
    )
    if hybrid_col
    else 0
)

pending_reviews = len(review)

reconciliation_rate = (
    matched / total * 100
    if total > 0
    else 0
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💳 RazorRecon AI")
    st.caption("AI Finance Controller")
    st.caption(
        "Reconcile • Investigate • Explain • Escalate • Review"
    )
    st.markdown("### Finance Control Architecture")

    st.markdown(
        """
        **Rules**  
        Authoritative

        **ML**  
        Supporting Signal

        **Gemini AI**  
        Explanation

        **Agent**  
        Decision Workflow

        **Human**  
        Final Review
        """
    )

    st.divider()

    st.markdown(
        """
        ### Safety Controls

        🟢 Rules remain authoritative

        🟡 ML cannot override rules

        🟡 Agent actions are simulated

        🔴 Human review is required
        """
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title"> 💳RazorRecon AI </div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI Finance Controller for Payment Reconciliation"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SAFETY BANNER
# ============================================================

st.markdown(
    """
    <div class="safe-box">
    <b>CONTROLLED EXECUTION</b><br>
    Deterministic rules are authoritative. ML is used only as a
    supporting signal. Agent actions and escalations are simulated.
    Human review remains required.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# KPI CARDS
# ============================================================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">TOTAL TRANSACTIONS</div>
            <div class="status-value">{number(total)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">MATCHED</div>
            <div class="status-value">{number(matched)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">EXCEPTIONS</div>
            <div class="status-value">{number(exceptions)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">RECONCILIATION RATE</div>
            <div class="status-value">{reconciliation_rate:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


k5, k6, k7, k8 = st.columns(4)

with k5:
    st.metric("🚨 Escalations", number(escalated))

with k6:
    st.metric("👤 Pending Reviews", number(pending_reviews))

with k7:
    st.metric("⚠️ ML Disagreements", number(disagreements))

with k8:
    # Financial difference is derived directly from the reconciliation
    # results; do not hard-code or manually add transaction amounts.
    financial = 0.0

    if (
        not reconciliation_results.empty
        and "difference" in reconciliation_results.columns
    ):
        financial = pd.to_numeric(
            reconciliation_results["difference"],
            errors="coerce",
        ).fillna(0).sum()

    st.metric(
        "💰 Financial Difference",
        money(financial),
    )


st.divider()


# ============================================================
# TABS
# ============================================================

overview_tab, transaction_tab, ml_tab, agent_tab, audit_tab = st.tabs(
    [
        "📊 Overview",
        "🔎 Transaction Investigation",
        "🧠 AI / ML",
        "🤖 Agent Workflow",
        "📋 Audit & Review",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # ARCHITECTURE FLOW
    # --------------------------------------------------------

    st.subheader("End-to-End Decision Flow")

    flow = st.columns(7)

    flow_data = [
        ("📥", "Transaction", "Input data"),
        ("⚙️", "Rules", "Authoritative"),
        ("🔴", "Exception", "Detection"),
        ("🧠", "AI / ML", "Support"),
        ("🤖", "Agent", "Decision"),
        ("👤", "Human", "Review"),
        ("📋", "Audit", "Traceability"),
    ]

    for column, (icon, title, subtitle) in zip(flow, flow_data):

        with column:

            st.markdown(
                f"""
                <div class="flow-box">
                    <div style="font-size:26px">{icon}</div>
                    <div class="flow-title">{title}</div>
                    <div class="flow-subtitle">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


    st.write("")


    # --------------------------------------------------------
    # STATUS / SEVERITY
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:
        counts = get_counts(
            dashboard,
            status_col
        )

        render_count_table(
            "transaction status",
            counts
        )

    with right:
        counts = get_counts(
            dashboard,
            severity_col
        )

        render_count_table(
            "severity",
            counts
        )


    # --------------------------------------------------------
    # EXCEPTION TYPES
    # --------------------------------------------------------

    exception_counts = get_counts(
        dashboard,
        exception_col
    )

    render_count_table(
        "exception distribution",
        exception_counts
    )


    # --------------------------------------------------------
    # TOP EXCEPTIONS
    # --------------------------------------------------------

    st.subheader("Top Exception Types")

    if not exception_analytics.empty:

        display_columns = [
            column
            for column in [
                "exception_type",
                "exception_count",
                "escalated_cases",
                "total_difference",
            ]
            if column in exception_analytics.columns
        ]

        if display_columns:

            top = exception_analytics[
                display_columns
            ].sort_values(
                by="exception_count",
                ascending=False
            )

            render_dataframe(top.head(10))


# ============================================================
# TRANSACTION INVESTIGATION
# ============================================================

ai_text = None
deterministic_text = None
recommended_action = None
explanation_source = None

with transaction_tab:

    st.markdown(
        '<div class="section-title">🔎 Transaction Investigation</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Select a transaction to trace its complete journey "
        "from reconciliation to human review."
    )

    transaction_col = find_column(
        dashboard,
        ["transaction_id"]
    )

    if transaction_col:

        transaction_ids = (
            dashboard[transaction_col]
            .dropna()
            .astype(str)
            .tolist()
        )

        selected_id = st.selectbox(
            "Transaction ID",
            transaction_ids,
        )

        selected = dashboard[
            dashboard[transaction_col].astype(str)
            == selected_id
        ]

        if not selected.empty:

            row = selected.iloc[0]

            ai_text = row.get("ai_explanation")
            recommended_action = row.get("recommended_action")
            explanation_source = row.get("explanation_source")

            if pd.isna(ai_text):
                ai_text = None
            if pd.isna(recommended_action):
                recommended_action = None
            if pd.isna(explanation_source):
                explanation_source = None

            if not ai_text and not ai_explanations.empty:
                explanation_id = find_column(
                    ai_explanations,
                    ["transaction_id"],
                )
                if explanation_id:
                    explanation_row = ai_explanations[
                        ai_explanations[explanation_id].astype(str)
                        == selected_id
                    ]
                    if not explanation_row.empty:
                        explanation = explanation_row.iloc[0]
                        deterministic_text = explanation.get(
                            "deterministic_explanation"
                        )
                        ai_text = explanation.get("ai_explanation")
                        if pd.isna(deterministic_text):
                            deterministic_text = None
                        if pd.isna(ai_text):
                            ai_text = None

            # ------------------------------------------------
            # HEADER
            # ------------------------------------------------

            st.divider()

            status = row.get("status", "N/A")
            exception_type = row.get(
                "exception_type",
                "N/A"
            )
            severity = row.get(
                "severity",
                "N/A"
            )
            hybrid_decision = row.get(
                "hybrid_decision",
                "N/A"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Status", str(status))
            c2.metric(
                "Exception Type",
                str(exception_type),
            )
            c3.metric(
                "Severity",
                str(severity),
            )
            c4.metric(
                "Hybrid Decision",
                str(hybrid_decision),
            )


            # ------------------------------------------------
            # TRANSACTION DETAILS
            # ------------------------------------------------

            st.subheader("Transaction Evidence")

            detail_columns = [
                column
                for column in [
                    "transaction_id",
                    "order_amount",
                    "payment_amount",
                    "bank_amount",
                    "settlement_amount",
                    "order_date",
                    "payment_date",
                    "settlement_date",
                    "payment_success",
                    "payment_failed",
                    "refund_amount",
                    "chargeback_amount",
                ]
                if column in selected.columns
            ]

            if detail_columns:

                details = selected[
                    detail_columns
                ].T.reset_index()

                details.columns = [
                    "Field",
                    "Value"
                ]

                details = details[
                    details["Value"].notna()
                ]

                render_dataframe(details)


            # ------------------------------------------------
            # AI EXPLANATION / AGENT INVESTIGATION REPORT
            # ------------------------------------------------

            # First try the exception explanation record for this transaction.
            ai_text = row.get("ai_explanation")
            deterministic_text = row.get("deterministic_explanation")
            recommended_action = row.get("recommended_action")
            explanation_source = row.get("explanation_source")

            def clean_value(value):
                if value is None or pd.isna(value):
                    return None
                value = str(value).strip()
                return value if value else None

            ai_text = clean_value(ai_text)
            deterministic_text = clean_value(deterministic_text)
            recommended_action = clean_value(recommended_action)
            explanation_source = clean_value(explanation_source)

            if not explanations.empty:
                exp_id = find_column(explanations, ["transaction_id"])

                if exp_id:
                    exp = explanations[
                        explanations[exp_id].astype(str).str.strip()
                        == str(selected_id).strip()
                    ]

                    if not exp.empty:
                        erow = exp.iloc[0]
                        ai_text = clean_value(erow.get("ai_explanation")) or ai_text
                        deterministic_text = (
                            clean_value(erow.get("deterministic_explanation"))
                            or deterministic_text
                        )
                        recommended_action = (
                            clean_value(erow.get("recommended_action"))
                            or recommended_action
                        )
                        explanation_source = (
                            clean_value(erow.get("explanation_source"))
                            or explanation_source
                        )

            # Second source: ai_explanations.csv.
            if not ai_text and not ai_explanations.empty:
                explanation_id = find_column(
                    ai_explanations,
                    ["transaction_id"],
                )

                if explanation_id:
                    explanation_rows = ai_explanations[
                        ai_explanations[explanation_id].astype(str).str.strip()
                        == str(selected_id).strip()
                    ]

                    if not explanation_rows.empty:
                        explanation = explanation_rows.iloc[0]

                        possible_ai_columns = [
                            "ai_explanation",
                            "explanation",
                            "generated_explanation",
                            "text",
                            "response",
                        ]

                        for column in possible_ai_columns:
                            if column in explanation_rows.columns:
                                candidate = clean_value(explanation.get(column))
                                if candidate:
                                    ai_text = candidate
                                    explanation_source = explanation_source or "AI"
                                    break

                        deterministic_text = (
                            clean_value(explanation.get("deterministic_explanation"))
                            or deterministic_text
                        )
                        recommended_action = (
                            clean_value(explanation.get("recommended_action"))
                            or recommended_action
                        )

            st.subheader("🤖 Agent Investigation Report")

            if ai_text:
                st.markdown(
                    dedent(f"""
                    <div style="padding:18px;border-radius:10px;
                                background:#172554;border:1px solid #2563eb;
                                margin-bottom:12px;">
                        <div style="font-size:14px;font-weight:600;
                                    margin-bottom:8px;">
                            🧠 AI Assessment
                        </div>
                        <div style="font-size:15px;line-height:1.6;">
                            {ai_text}
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )
                st.caption(f"Explanation source: {explanation_source or 'AI'}")

            elif deterministic_text:
                st.markdown(
                    dedent(f"""
                    <div style="padding:18px;border-radius:10px;
                                background:#3f2a05;border:1px solid #f59e0b;
                                margin-bottom:12px;">
                        <div style="font-size:14px;font-weight:600;
                                    margin-bottom:8px;">
                            📋 Deterministic Investigation
                        </div>
                        <div style="font-size:15px;line-height:1.6;">
                            {deterministic_text}
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )
                st.caption(
                    "AI explanation unavailable; showing deterministic explanation."
                )

            else:
                st.warning(
                    "No investigation explanation was found for "
                    f"transaction {selected_id}."
                )

            if recommended_action:
                st.markdown(
                    dedent(f"""
                    <div style="padding:16px;border-radius:10px;
                                background:#052e1b;border:1px solid #16a34a;
                                margin-top:8px;">
                        <div style="font-size:14px;font-weight:600;
                                    margin-bottom:7px;">
                            🎯 Recommended Next Step
                        </div>
                        <div style="font-size:15px;line-height:1.5;">
                            {recommended_action}
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )

            # ------------------------------------------------
            # ML CONFIDENCE / SUPPORTING SIGNAL
            # ------------------------------------------------

            st.subheader("🤖 ML Supporting Signal")

            ml_row = pd.DataFrame()

            if not ml_agent.empty:
                ml_id = find_column(ml_agent, ["transaction_id"])
                if ml_id:
                    ml_row = ml_agent[
                        ml_agent[ml_id].astype(str).str.strip()
                        == str(selected_id).strip()
                    ]

            if ml_row.empty and not dashboard.empty:
                dashboard_id = find_column(dashboard, ["transaction_id"])
                if dashboard_id:
                    dashboard_ml_row = dashboard[
                        dashboard[dashboard_id].astype(str).str.strip()
                        == str(selected_id).strip()
                    ]
                    if not dashboard_ml_row.empty:
                        ml_columns = [
                            column for column in [
                                "ml_predicted_exception_type",
                                "ml_exception_probability",
                                "ml_agrees_with_rule_status",
                                "ml_agrees_with_rule_exception_type",
                            ]
                            if column in dashboard_ml_row.columns
                        ]
                        if ml_columns:
                            ml_row = dashboard_ml_row[
                                [dashboard_id] + ml_columns
                            ]

            if not ml_row.empty:
                mrow = ml_row.iloc[0]
                probability = clean_value(
                    mrow.get("ml_exception_probability")
                )

                if probability is not None:
                    try:
                        confidence = max(
                            0.0,
                            min(1.0, float(probability))
                        )
                        st.progress(confidence)
                        st.caption(
                            f"ML exception confidence: {confidence:.1%}"
                        )
                    except (TypeError, ValueError):
                        pass

                ml_display = [
                    column for column in [
                        "ml_predicted_exception_type",
                        "ml_exception_probability",
                        "ml_agrees_with_rule_status",
                        "ml_agrees_with_rule_exception_type",
                    ]
                    if column in ml_row.columns
                ]

                if ml_display:
                    render_dataframe(ml_row[ml_display])
                else:
                    st.info(
                        "ML supporting signal fields are not "
                        "available for this transaction."
                    )
            else:
                st.info(
                    "No ML supporting signal is available "
                    f"for transaction {selected_id}."
                )

            # ------------------------------------------------
            # EXCEPTION
            # ------------------------------------------------

            st.subheader("🚨 Exception")

            exception_display = []

            for column in [
                "exception_type",
                "severity",
                "difference",
                "reason",
            ]:
                if column in row.index:
                    value = row.get(column)
                    if pd.notna(value):
                        exception_display.append(
                            {"Field": column, "Value": value}
                        )

            if exception_display:
                render_dataframe(pd.DataFrame(exception_display))
            else:
                st.info(
                    "No exception details are available "
                    f"for transaction {selected_id}."
                )

            # ------------------------------------------------
            # AGENT DECISION
            # ------------------------------------------------

            st.subheader("🤖 Agent Decision")

            agent_row = pd.DataFrame()

            if not agent.empty:
                agent_id = find_column(agent, ["transaction_id"])
                if agent_id:
                    agent_row = agent[
                        agent[agent_id].astype(str).str.strip()
                        == str(selected_id).strip()
                    ]

            if not agent_row.empty:
                agent_columns = [
                    column for column in [
                        "agent_decision",
                        "escalation_required",
                        "action_type",
                        "action_status",
                        "requires_human",
                        "execution_mode",
                        "agent_decision_reason",
                    ]
                    if column in agent_row.columns
                ]

                if agent_columns:
                    render_dataframe(agent_row[agent_columns])
                else:
                    st.info("Agent decision fields are not available.")
            else:
                st.info(
                    "No agent decision is available "
                    f"for transaction {selected_id}."
                )

            # ------------------------------------------------
            # ESCALATION
            # ------------------------------------------------

            st.subheader("🚨 Escalation")

            escalation_row = pd.DataFrame()

            if not escalations.empty:
                escalation_id = find_column(
                    escalations,
                    ["transaction_id"]
                )
                if escalation_id:
                    escalation_row = escalations[
                        escalations[escalation_id]
                        .astype(str).str.strip()
                        == str(selected_id).strip()
                    ]

            if not escalation_row.empty:
                escalation_columns = [
                    column for column in [
                        "transaction_id",
                        "exception_type",
                        "severity",
                        "difference",
                        "priority",
                        "resolution_category",
                        "next_step",
                        "escalation_required",
                        "escalation_priority",
                        "escalation_reason",
                        "escalation_status",
                        "review_owner",
                        "escalation_mode",
                    ]
                    if column in escalation_row.columns
                ]

                if escalation_columns:
                    render_dataframe(
                        escalation_row[escalation_columns]
                    )
                else:
                    st.info("Escalation details are not available.")
            else:
                st.info(
                    "No escalation case is available "
                    f"for transaction {selected_id}."
                )

            # ------------------------------------------------
            # HUMAN REVIEW
            # ------------------------------------------------

            st.subheader("👤 Human Review")

            review_row = pd.DataFrame()

            if not review.empty:
                review_id = find_column(
                    review,
                    ["transaction_id"]
                )
                if review_id:
                    review_row = review[
                        review[review_id]
                        .astype(str).str.strip()
                        == str(selected_id).strip()
                    ]

            if not review_row.empty:
                review_columns = [
                    column for column in [
                        "review_case_id",
                        "event_id",
                        "audit_id",
                        "transaction_id",
                        "event_type",
                        "event_actor",
                        "event_source",
                        "event_status",
                        "exception_type",
                        "severity",
                        "difference",
                        "hybrid_decision",
                        "agent_decision",
                        "action_type",
                        "action_status",
                        "execution_mode",
                        "escalation_required",
                        "escalation_priority",
                        "escalation_status",
                        "review_owner",
                        "review_priority",
                        "queue_status",
                        "review_status",
                        "requires_human_review",
                        "review_mode",
                    ]
                    if column in review_row.columns
                ]

                if review_columns:
                    render_dataframe(review_row[review_columns])
                else:
                    st.info(
                        "Human review details are not available."
                    )
            else:
                st.info(
                    "No human review case is available "
                    f"for transaction {selected_id}."
                )


# ============================================================
# AI / ML TAB
# ============================================================

with ml_tab:

    st.markdown(
        '<div class="section-title">🧠 AI / ML Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Hybrid Intelligence

        **Deterministic Rules** → authoritative financial decision

        **Random Forest ML** → supporting prediction

        **Gemini AI** → natural-language explanation

        **Agent** → controlled workflow decision

        ML is deliberately prevented from overriding deterministic
        reconciliation results.
        """
    )

    st.divider()

    st.subheader("Hybrid Decision Distribution")

    hybrid_counts = get_counts(
        hybrid,
        find_column(hybrid, ["hybrid_decision"])
    )

    render_count_table("hybrid distribution", hybrid_counts)

    st.subheader("ML / Agent Analytics")

    if not ml_agent.empty:
        render_dataframe(ml_agent)
    else:
        st.info("No ML / agent analytics data available.")

    st.subheader("Model Performance")

    a, b, c, d = st.columns(4)
    a.metric("Binary Accuracy", "96.00%")
    b.metric("Binary Precision", "100.00%")
    c.metric("Binary Recall", "80.00%")
    d.metric("Binary F1", "88.89%")

    a, b, c, d = st.columns(4)
    a.metric("Multiclass Accuracy", "96.00%")
    b.metric("Weighted F1", "94.47%")
    c.metric("Macro F1", "78.97%")
    d.metric("ML Disagreements", number(disagreements))


# ============================================================
# AGENT WORKFLOW TAB
# ============================================================

with agent_tab:

    st.markdown(
        '<div class="section-title">🤖 Agent Workflow</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        The agent converts reconciliation exceptions into a controlled
        operational workflow.

        **Exception → Decision → Action → Escalation → Human Review**

        Financial records are not directly modified by the agent.
        """
    )

    st.divider()

    st.subheader("Agent Decision Distribution")

    agent_decision_counts = get_counts(
        agent,
        find_column(agent, ["agent_decision"])
    )

    render_count_table("agent decisions", agent_decision_counts)

    st.subheader("Action Distribution")

    action_counts = get_counts(
        actions,
        find_column(actions, ["action_type"])
    )

    render_count_table("actions", action_counts)

    st.subheader("Escalation Cases")

    if not escalations.empty:
        escalation_display_columns = [
            column for column in [
                "transaction_id",
                "exception_type",
                "severity",
                "difference",
                "priority",
                "resolution_category",
                "next_step",
                "escalation_required",
                "escalation_priority",
                "escalation_reason",
                "escalation_status",
                "review_owner",
                "escalation_mode",
            ]
            if column in escalations.columns
        ]

        if escalation_display_columns:
            render_dataframe(
                escalations[escalation_display_columns]
            )
        else:
            render_dataframe(escalations)
    else:
        st.info("No escalation cases available.")

    st.subheader("Escalation Priority")

    escalation_priority_col = find_column(
        escalations,
        ["escalation_priority", "priority"]
    )

    if escalation_priority_col and not escalations.empty:
        priority_counts = get_counts(
            escalations,
            escalation_priority_col
        )
        render_count_table(
            "escalation priority",
            priority_counts
        )
    else:
        st.info("No escalation priority data available.")


# ============================================================
# AUDIT TAB
# ============================================================

with audit_tab:

    st.markdown(
        '<div class="section-title">📋 Audit & Human Review</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Every exception is traceable through:

        **Exception → Agent Decision → Action → Escalation → Human Review → Audit**
        """
    )

    st.divider()

    a, b, c = st.columns(3)

    a.metric(
        "Audit Records",
        number(len(audit))
    )

    b.metric(
        "Audit Events",
        number(len(audit))
    )

    c.metric(
        "Review Cases",
        number(len(review))
    )


    st.subheader("Audit Events")

    if not audit.empty:

        render_dataframe(audit)


    st.subheader("Human Review Queue")

    if not review.empty:

        render_dataframe(review)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Reconciliation Agent • "
    "Rules = Authoritative • "
    "ML = Supporting Signal • "
    "AI = Explanation • "
    "Actions = Simulated • "
    "Human Review = Required"
)