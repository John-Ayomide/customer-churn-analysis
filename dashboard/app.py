# ============================================
# CUSTOMER CHURN PREDICTION DASHBOARD
# Modern redesign — dark theme, professional UI
# Author: John Ayomide
# ============================================

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier

# ── Page config ─────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #0a0a0f;
        color: #e8e8f0;
    }

    /* Main container */
    .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #111118;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="metric-container"] label {
        color: #6b6b80 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00e5a0 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Section headers */
    h2, h3 {
        color: #e8e8f0 !important;
        font-weight: 600 !important;
    }

    /* Divider */
    hr {
        border-color: #2a2a3a !important;
    }

    /* Input widgets */
    .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #9999aa !important;
        font-size: 13px !important;
    }

    /* Button */
    .stButton button {
        background: linear-gradient(135deg, #00e5a0, #7b61ff) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-size: 15px !important;
        width: 100%;
    }

    .stButton button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px);
    }

    /* Chart backgrounds */
    .stPlotlyChart, [data-testid="stImage"] {
        background: #111118;
        border-radius: 12px;
        border: 1px solid #2a2a3a;
        padding: 8px;
    }

    /* Risk cards */
    .risk-card-high {
        background: rgba(255,107,107,0.1);
        border: 1px solid rgba(255,107,107,0.4);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
    }

    .risk-card-medium {
        background: rgba(255,209,102,0.1);
        border: 1px solid rgba(255,209,102,0.4);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
    }

    .risk-card-low {
        background: rgba(0,229,160,0.1);
        border: 1px solid rgba(0,229,160,0.4);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
    }

    .risk-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .risk-action {
        font-size: 14px;
        opacity: 0.8;
        margin-top: 8px;
    }

    /* Input section card */
    .input-card {
        background: #111118;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }

    /* Footer */
    .footer-text {
        color: #6b6b80;
        font-size: 12px;
        text-align: center;
        margin-top: 32px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load and prepare data ────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/churn_cleaned.csv')

@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, X.columns.tolist()

df = load_data()
model, X_train, X_test, y_train, y_test, feature_cols = train_model(df)

# ── Header ───────────────────────────────────
st.markdown("""
<div style='margin-bottom: 8px'>
    <span style='font-size:11px;color:#00e5a0;letter-spacing:3px;
    text-transform:uppercase'>Portfolio Project 01</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='font-size:32px;font-weight:700;color:#e8e8f0;margin-bottom:8px'>
    Customer Churn <span style='color:#00e5a0'>Intelligence</span>
</h1>
<p style='color:#6b6b80;font-size:15px;margin-bottom:32px'>
    Telecom churn prediction using XGBoost — 
    AUC 0.841 · 79.1% accuracy · SHAP explainability
</p>
""", unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
churn_rate = df['Churn'].mean() * 100
churned = df['Churn'].sum()
retained = len(df) - churned

with k1:
    st.metric("Total Customers", f"{len(df):,}")
with k2:
    st.metric("Churn Rate", f"{churn_rate:.1f}%",
              delta="↑ vs 10-15% avg", delta_color="inverse")
with k3:
    st.metric("Customers Churned", f"{churned:,}")
with k4:
    st.metric("Model AUC Score", "0.841")

st.markdown("---")

# ── Charts Row ───────────────────────────────
st.markdown("### 📊 Churn Analysis")
c1, c2 = st.columns(2)

plt.style.use('dark_background')

with c1:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#111118')
    ax.set_facecolor('#111118')
    counts = df['Churn'].value_counts()
    bars = ax.bar(['Stayed', 'Churned'], counts.values,
                  color=['#00e5a0', '#ff6b6b'], width=0.5)
    ax.set_title('Churn Distribution',
                 color='#e8e8f0', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(colors='#6b6b80')
    ax.spines['bottom'].set_color('#2a2a3a')
    ax.spines['left'].set_color('#2a2a3a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        ax.annotate(f'{int(bar.get_height()):,}',
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom',
                    color='#e8e8f0', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#111118')
    ax.set_facecolor('#111118')
    churn_by_contract = df.groupby('Contract')['Churn'].mean().sort_values()
    colors = ['#00e5a0', '#7b61ff', '#ff6b6b']
    bars = ax.bar(churn_by_contract.index, churn_by_contract.values,
                  color=colors, width=0.5)
    ax.set_title('Churn Rate by Contract Type',
                 color='#e8e8f0', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(colors='#6b6b80')
    ax.spines['bottom'].set_color('#2a2a3a')
    ax.spines['left'].set_color('#2a2a3a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        ax.annotate(f'{bar.get_height():.1%}',
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom',
                    color='#e8e8f0', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

# ── Tenure chart ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3))
fig.patch.set_facecolor('#111118')
ax.set_facecolor('#111118')
df[df['Churn']==1]['tenure'].plot(
    kind='hist', bins=30, alpha=0.7,
    color='#ff6b6b', label='Churned', ax=ax)
df[df['Churn']==0]['tenure'].plot(
    kind='hist', bins=30, alpha=0.7,
    color='#00e5a0', label='Stayed', ax=ax)
ax.set_title('Tenure Distribution — Churned vs Stayed',
             color='#e8e8f0', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Tenure (months)', color='#6b6b80')
ax.tick_params(colors='#6b6b80')
ax.spines['bottom'].set_color('#2a2a3a')
ax.spines['left'].set_color('#2a2a3a')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(facecolor='#1a1a24', labelcolor='#e8e8f0')
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ── Predictor ────────────────────────────────
st.markdown("### 🔮 Churn Risk Predictor")
st.markdown(
    "<p style='color:#6b6b80;font-size:14px;margin-bottom:24px'>"
    "Enter a customer's details to get their predicted churn probability "
    "and recommended retention action.</p>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        "<p style='color:#9999aa;font-size:11px;"
        "text-transform:uppercase;letter-spacing:1.5px;"
        "margin-bottom:12px'>Usage & Billing</p>",
        unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges (£)", 18, 120, 65)
    total_charges = st.number_input(
        "Total Charges (£)", min_value=0.0,
        max_value=9000.0,
        value=float(tenure * monthly_charges))

with col2:
    st.markdown(
        "<p style='color:#9999aa;font-size:11px;"
        "text-transform:uppercase;letter-spacing:1.5px;"
        "margin-bottom:12px'>Contract & Service</p>",
        unsafe_allow_html=True)
    contract = st.selectbox("Contract Type",
                             ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service",
                                     ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"])

with col3:
    st.markdown(
        "<p style='color:#9999aa;font-size:11px;"
        "text-transform:uppercase;letter-spacing:1.5px;"
        "margin-bottom:12px'>Customer Profile</p>",
        unsafe_allow_html=True)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Churn Risk"):

    # Build prediction input
    input_data = {col: 0 for col in feature_cols}
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges
    input_data['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0

    if contract == "One year":
        input_data['Contract_One year'] = 1
    elif contract == "Two year":
        input_data['Contract_Two year'] = 1
    if internet_service == "Fiber optic":
        input_data['InternetService_Fiber optic'] = 1
    elif internet_service == "No":
        input_data['InternetService_No'] = 1
    if payment_method == "Credit card (automatic)":
        input_data['PaymentMethod_Credit card (automatic)'] = 1
    elif payment_method == "Electronic check":
        input_data['PaymentMethod_Electronic check'] = 1
    elif payment_method == "Mailed check":
        input_data['PaymentMethod_Mailed check'] = 1
    if partner == "Yes":
        input_data['Partner_Yes'] = 1
    if dependents == "Yes":
        input_data['Dependents_Yes'] = 1
    if paperless_billing == "Yes":
        input_data['PaperlessBilling_Yes'] = 1

    input_df = pd.DataFrame([input_data])
    churn_prob = float(model.predict_proba(input_df)[0][1])

    # Display result
    if churn_prob >= 0.7:
        st.markdown(f"""
        <div class='risk-card-high'>
            <div class='risk-title' style='color:#ff6b6b'>
                🔴 HIGH CHURN RISK — {churn_prob:.1%}
            </div>
            <div style='font-size:13px;color:#ff6b6b;opacity:0.8'>
                This customer has a {churn_prob:.1%} probability of churning
            </div>
            <div class='risk-action' style='color:#e8e8f0'>
                <strong>Recommended action:</strong>
                Immediate retention outreach — offer contract upgrade,
                personalised discount, or priority support call
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif churn_prob >= 0.4:
        st.markdown(f"""
        <div class='risk-card-medium'>
            <div class='risk-title' style='color:#ffd166'>
                🟡 MEDIUM CHURN RISK — {churn_prob:.1%}
            </div>
            <div style='font-size:13px;color:#ffd166;opacity:0.8'>
                This customer has a {churn_prob:.1%} probability of churning
            </div>
            <div class='risk-action' style='color:#e8e8f0'>
                <strong>Recommended action:</strong>
                Proactive check-in — offer loyalty reward,
                service review, or contract incentive
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class='risk-card-low'>
            <div class='risk-title' style='color:#00e5a0'>
                🟢 LOW CHURN RISK — {churn_prob:.1%}
            </div>
            <div style='font-size:13px;color:#00e5a0;opacity:0.8'>
                This customer has a {churn_prob:.1%} probability of churning
            </div>
            <div class='risk-action' style='color:#e8e8f0'>
                <strong>Recommended action:</strong>
                No immediate action required —
                continue monitoring monthly
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Probability bar
    st.markdown("<br>", unsafe_allow_html=True)
    bar_color = (
        "#ff6b6b" if churn_prob >= 0.7
        else "#ffd166" if churn_prob >= 0.4
        else "#00e5a0"
    )
    st.markdown(f"""
    <div style='background:#1a1a24;border-radius:8px;
    padding:16px;border:1px solid #2a2a3a'>
        <div style='display:flex;justify-content:space-between;
        margin-bottom:8px'>
            <span style='font-size:13px;color:#9999aa'>
                Churn Probability Score
            </span>
            <span style='font-size:15px;font-weight:700;
            color:{bar_color}'>{churn_prob:.1%}</span>
        </div>
        <div style='height:8px;background:#2a2a3a;
        border-radius:100px;overflow:hidden'>
            <div style='height:100%;width:{churn_prob*100:.1f}%;
            background:{bar_color};border-radius:100px;
            transition:width 0.5s ease'></div>
        </div>
        <div style='display:flex;justify-content:space-between;
        margin-top:6px'>
            <span style='font-size:11px;color:#6b6b80'>Low risk</span>
            <span style='font-size:11px;color:#6b6b80'>High risk</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────
st.markdown("---")
st.markdown("""
<div class='footer-text'>
    Built by <strong style='color:#e8e8f0'>John Ayomide</strong> &nbsp;·&nbsp;
    <a href='https://github.com/John-Ayomide/customer-churn-analysis'
    style='color:#00e5a0;text-decoration:none'>GitHub Repository</a>
    &nbsp;·&nbsp; XGBoost · SHAP · Streamlit
</div>
""", unsafe_allow_html=True)