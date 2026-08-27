import streamlit as st
import pandas as pd
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Reconciliation Agent",
    page_icon="🤖",
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


# ============================================================
# MAIN COLUMNS
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

    st.title("🤖 AI Reconciliation")

    st.markdown("### System Architecture")

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
    '<div class="main-title">🤖 AI Reconciliation Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Intelligent Transaction Reconciliation & Exception Management"
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

    financial = 0

    if "total_difference" in exception_analytics.columns:
        financial = exception_analytics["total_difference"].sum()

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

        st.subheader("Transaction Status")

        counts = get_counts(
            dashboard,
            status_col
        )

        if not counts.empty:
            st.bar_chart(counts)

    with right:

        st.subheader("Severity")

        counts = get_counts(
            dashboard,
            severity_col
        )

        if not counts.empty:
            st.bar_chart(counts)


    # --------------------------------------------------------
    # EXCEPTION TYPES
    # --------------------------------------------------------

    st.subheader("Exception Distribution")

    exception_counts = get_counts(
        dashboard,
        exception_col
    )

    if not exception_counts.empty:
        st.bar_chart(exception_counts)


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

            st.dataframe(
                top.head(10),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# TRANSACTION INVESTIGATION
# ============================================================

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

                st.dataframe(
                    details,
                    use_container_width=True,
                    hide_index=True,
                )


            # ------------------------------------------------
            # AI EXPLANATION
            # ------------------------------------------------

            # ------------------------------------------------
# AI EXPLANATION
# ------------------------------------------------

st.subheader("🧠 AI Explanation")

explanation_found = False
ai_text = None
deterministic_text = None
recommended_action = None
explanation_source = None


# ========================================================
# FIRST: SEARCH exception_explanations.csv
# ========================================================

if not explanations.empty:

    exp_id = find_column(
        explanations,
        ["transaction_id"]
    )

    if exp_id:

        exp = explanations[
            explanations[exp_id].astype(str).str.strip()
            == str(selected_id).strip()
        ]

        if not exp.empty:

            erow = exp.iloc[0]

            if "ai_explanation" in exp.columns:
                value = erow["ai_explanation"]

                if pd.notna(value) and str(value).strip():
                    ai_text = str(value).strip()

            if "deterministic_explanation" in exp.columns:
                value = erow["deterministic_explanation"]

                if pd.notna(value) and str(value).strip():
                    deterministic_text = str(value).strip()

            if "recommended_action" in exp.columns:
                value = erow["recommended_action"]

                if pd.notna(value) and str(value).strip():
                    recommended_action = str(value).strip()

            if "explanation_source" in exp.columns:
                value = erow["explanation_source"]

                if pd.notna(value) and str(value).strip():
                    explanation_source = str(value).strip()

            explanation_found = True


# ========================================================
# SECOND: SEARCH ai_explanations.csv
# ========================================================

if not ai_text and not ai_explanations.empty:

    ai_id = find_column(
        ai_explanations,
        ["transaction_id"]
    )

    if ai_id:

        ai_row = ai_explanations[
            ai_explanations[ai_id].astype(str).str.strip()
            == str(selected_id).strip()
        ]

        if not ai_row.empty:

            arow = ai_row.iloc[0]

            possible_ai_columns = [
                "ai_explanation",
                "explanation",
                "generated_explanation",
                "text",
                "response",
            ]

            for column in possible_ai_columns:

                if column in ai_row.columns:

                    value = arow[column]

                    if pd.notna(value) and str(value).strip():

                        ai_text = str(value).strip()
                        explanation_source = "AI"
                        explanation_found = True
                        break


# ========================================================
# DISPLAY AI EXPLANATION
# ========================================================

if ai_text:

    st.info(ai_text)

    st.caption(
        f"Explanation source: "
        f"{explanation_source or 'AI'}"
    )


# ========================================================
# DISPLAY DETERMINISTIC EXPLANATION IF AI MISSING
# ========================================================

elif deterministic_text:

    st.info(
        deterministic_text
    )

    st.caption(
        "AI explanation unavailable; "
        "showing deterministic explanation."
    )


# ========================================================
# NOTHING FOUND
# ========================================================

else:

    st.warning(
        "No explanation text was found for "
        f"transaction {selected_id}."
    )


# ========================================================
# RECOMMENDED ACTION
# ========================================================

if recommended_action:

    st.markdown("**Recommended Action**")

    st.success(
        recommended_action
    )

    # ------------------------------------------------
    # ML SIGNAL
    # ------------------------------------------------

    st.subheader("🤖 ML Supporting Signal")

    ml_fields = [
        "ml_predicted_exception",
        "ml_predicted_exception_type",
        "ml_exception_probability",
        "ml_agrees_with_rule_status",
        "ml_agrees_with_rule_exception_type",
    ]

    ml_rows = []

    for field in ml_fields:

        if field in selected.columns:

            value = row[field]

            if pd.notna(value):

                ml_rows.append(
                    [field, value]
                )

    if ml_rows:

        st.dataframe(
            pd.DataFrame(
                ml_rows,
                columns=[
                    "ML Field",
                    "Value"
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No ML fields available for this transaction."
        )


    # ------------------------------------------------
    # AGENT DECISION
    # ------------------------------------------------

    st.subheader("🤖 Agent Decision")

    agent_fields = [
        "agent_decision",
        "resolution_category",
        "escalation_required",
        "action_type",
        "action_status",
        "requires_human",
        "execution_mode",
    ]

    agent_rows = []

    for field in agent_fields:

        if field in selected.columns:

            value = row[field]

            if pd.notna(value):

                agent_rows.append(
                    [field, value]
                )

    if agent_rows:

        st.dataframe(
            pd.DataFrame(
                agent_rows,
                columns=[
                    "Agent Field",
                    "Value"
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )


    # ------------------------------------------------
    # ESCALATION
    # ------------------------------------------------

    st.subheader("🚨 Escalation")

    if not escalations.empty:

        esc_id = find_column(
            escalations,
            ["transaction_id"]
        )

        if esc_id:

            esc = escalations[
                escalations[esc_id].astype(str)
                == selected_id
            ]

            if not esc.empty:

                st.dataframe(
                    esc,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.success(
                    "No escalation case created."
                )

    # ------------------------------------------------
    # REVIEW STATUS
    # ------------------------------------------------

    st.subheader("👤 Human Review")

    if not review.empty:

        review_id = find_column(
            review,
            ["transaction_id"]
        )

        if review_id:

            review_case = review[
                review[review_id].astype(str)
                == selected_id
            ]

            if not review_case.empty:

                st.dataframe(
                    review_case,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No review case found."
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

        ML is deliberately prevented from overriding deterministic
        reconciliation results.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    st.subheader("Hybrid Decision Distribution")

    hybrid_counts = get_counts(
        hybrid,
        find_column(
            hybrid,
            ["hybrid_decision"]
        )
    )

    if not hybrid_counts.empty:
        st.bar_chart(hybrid_counts)

        st.dataframe(
            hybrid_counts.rename(
                "Cases"
            ).reset_index(),
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # ML / AGENT ANALYTICS
    # --------------------------------------------------------

    st.subheader("ML / Agent Analytics")

    if not ml_agent.empty:

        st.dataframe(
            ml_agent,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # KEY ML NUMBERS
    # --------------------------------------------------------

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
# AGENT TAB
# ============================================================

with agent_tab:

    st.markdown(
        '<div class="section-title">🤖 Agent Workflow</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        The agent converts exception information into a controlled
        workflow. It does **not** directly modify financial records.
        Actions are simulated and human review is required.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # DECISIONS
    # --------------------------------------------------------

    st.subheader("Agent Decision Distribution")

    decision_counts = get_counts(
        agent,
        find_column(
            agent,
            ["agent_decision"]
        )
    )

    if not decision_counts.empty:

        st.bar_chart(decision_counts)

        st.dataframe(
            decision_counts.rename(
                "Cases"
            ).reset_index(),
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # ACTIONS
    # --------------------------------------------------------

    st.subheader("Action Distribution")

    action_counts = get_counts(
        actions,
        find_column(
            actions,
            ["action_type"]
        )
    )

    if not action_counts.empty:

        st.bar_chart(action_counts)


    # --------------------------------------------------------
    # ESCALATION
    # --------------------------------------------------------

    st.subheader("Escalation Cases")

    if not escalations.empty:

        priority_col = find_column(
            escalations,
            ["escalation_priority"]
        )

        if priority_col:

            priority_counts = get_counts(
                escalations,
                priority_col
            )

            st.bar_chart(priority_counts)

        st.dataframe(
            escalations,
            use_container_width=True,
            hide_index=True,
        )


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

        st.dataframe(
            audit,
            use_container_width=True,
            hide_index=True,
        )


    st.subheader("Human Review Queue")

    if not review.empty:

        st.dataframe(
            review,
            use_container_width=True,
            hide_index=True,
        )


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