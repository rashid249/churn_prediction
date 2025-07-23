# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
sns.set(style='whitegrid')

# ---------------------
# Page Setup
# ---------------------
st.set_page_config(page_title="📊 Telecom Churn Dashboard", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# ---------------------
# Load Data & Model
# ---------------------
@st.cache_data
def load_data():
    return pd.read_csv('churn_dataset.csv')

@st.cache_resource
def load_advanced_model():
    with open('advanced_churn_model.pkl', 'rb') as f:
        model, scaler, columns = pickle.load(f)
    return model, scaler, columns

data = load_data()
model, scaler, model_columns = load_advanced_model()

# ---------------------
# Sidebar: Input Form
# ---------------------
st.sidebar.header("🔮 Predict Customer Churn")
st.sidebar.markdown("Fill in the customer details:")

tenure = st.sidebar.slider('Tenure (months)', 0, 100, 12)
monthly = st.sidebar.number_input('Monthly Charges', 0.0, 200.0, 70.0)
total = st.sidebar.number_input('Total Charges', 0.0, 10000.0, 2500.0)
contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
payment = st.sidebar.selectbox('Payment Method', [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
internet = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

if st.sidebar.button("🚀 Predict Now"):
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total],
        f'Contract_{contract}': [1],
        f'PaymentMethod_{payment}': [1],
        f'InternetService_{internet}': [1]
    })
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_columns]
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader("🎯 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ **Likely to churn!**\nProbability: **{prob:.1f}%**")
    else:
        st.success(f"✅ **Not likely to churn.**\nProbability: **{100 - prob:.1f}%**")

    # Feature importance
    st.subheader("📊 Top 5 Feature Importances")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'feature': model_columns, 'importance': importances})
    feat_df = feat_df.sort_values('importance', ascending=False).head(5)

    fig, ax = plt.subplots()
    bars = ax.barh(feat_df['feature'], feat_df['importance'], color=plt.cm.viridis(feat_df['importance']))
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    st.pyplot(fig)

# ---------------------
# Main App Title & Tabs
# ---------------------
st.title("📊 Telecom Customer Churn Dashboard")
st.caption("Explore churn patterns, predict churn, and gain business insights.")

churn_rate = (data['Churn'].value_counts(normalize=True) * 100).get('Yes', 0)
st.metric("📉 Overall Churn Rate", f"{churn_rate:.2f} %")

tab_viz, tab_info = st.tabs(["📈 Analysis & Insights", "📘 About Model"])

with tab_viz:
    st.subheader("✅ Churn Distribution")
    churn_counts = data['Churn'].value_counts()
    fig, ax = plt.subplots()
    bars = ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B', '#4ECDC4'])
    ax.bar_label(bars)
    st.pyplot(fig)

    st.subheader("📑 Churn by Contract Type")
    churn_rate_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0)*100
    fig, ax = plt.subplots()
    bars = ax.bar(churn_rate_contract.index, churn_rate_contract.values, color='#ffa600')
    ax.bar_label(bars, fmt='%.1f%%')
    ax.set_ylabel('Churn Rate (%)')
    st.pyplot(fig)

    st.subheader("💳 Churn by Payment Method")
    churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0)*100
    churn_rate_payment = churn_rate_payment.sort_values(ascending=False)
    fig, ax = plt.subplots()
    bars = ax.barh(churn_rate_payment.index, churn_rate_payment.values, color='#00b4d8')
    ax.bar_label(bars, fmt='%.1f%%')
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    st.markdown("""
    - 🚩 **Highest churn** for **month-to-month** contracts & **electronic check** payments.
    - 📈 Higher **monthly & total charges** are linked to churn.
    - 🕒 Short-tenure customers are more likely to churn.
    """)

with tab_info:
    st.subheader("📘 Model Details")
    st.markdown(f"""
    - **Model Type**: `{type(model).__name__}`
    - **Input Features**: `{len(model_columns)}`
    - **Scaler Used**: `{type(scaler).__name__}`
    - **Feature Columns**:
    """)
    st.code(", ".join(model_columns[:10]) + ("..." if len(model_columns) > 10 else ""))

st.markdown("---")  # ✅ Fixed the syntax error

