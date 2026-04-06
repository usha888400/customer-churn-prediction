import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Customer Churn Prediction System")
st.markdown("**AI-powered system to predict which customers are likely to leave**")

@st.cache_data
def load_and_train():
    np.random.seed(42)
    n = 1000
    data = {
        'age': np.random.randint(18, 70, n),
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_charges': np.random.uniform(20, 120, n),
        'total_charges': np.random.uniform(100, 8000, n),
        'num_products': np.random.randint(1, 5, n),
        'has_internet': np.random.choice([1, 0], n),
        'contract_type': np.random.choice([0, 1, 2], n),
        'payment_method': np.random.choice([0, 1, 2], n),
        'support_calls': np.random.randint(0, 10, n),
    }
    df = pd.DataFrame(data)
    churn_prob = (
        (df['tenure_months'] < 12) * 0.3 +
        (df['monthly_charges'] > 80) * 0.2 +
        (df['contract_type'] == 0) * 0.3 +
        (df['support_calls'] > 5) * 0.2
    )
    df['churn'] = (np.random.random(n) < churn_prob).astype(int)
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return df, model, X, accuracy

df, model, X, accuracy = load_and_train()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", len(df))
with col2:
    st.metric("Churn Rate", f"{df['churn'].mean():.1%}")
with col3:
    st.metric("Model Accuracy", f"{accuracy:.1%}")
with col4:
    st.metric("Avg Monthly Charges", f"${df['monthly_charges'].mean():.0f}")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Churn Distribution")
    fig1 = px.pie(df, names='churn', title='Churn vs No Churn',
                  color_discrete_map={0: 'green', 1: 'red'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📈 Monthly Charges vs Churn")
    fig2 = px.histogram(df, x='monthly_charges', color='churn',
                        barmode='overlay',
                        color_discrete_map={0: 'green', 1: 'red'})
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("🎯 Predict Individual Customer")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 18, 70, 35)
    tenure = st.number_input("Tenure (months)", 1, 72, 12)
    monthly = st.number_input("Monthly Charges ($)", 20, 120, 65)
with col2:
    total = st.number_input("Total Charges ($)", 100, 8000, 1000)
    products = st.number_input("Number of Products", 1, 5, 2)
    calls = st.number_input("Support Calls", 0, 10, 2)
with col3:
    internet = st.selectbox("Has Internet", ['Yes', 'No'])
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    payment = st.selectbox("Payment Method", ['Credit card', 'Bank transfer', 'Electronic check'])

if st.button("🔮 Predict Churn", type="primary"):
    internet_enc = 1 if internet == 'Yes' else 0
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_map = {'Credit card': 0, 'Bank transfer': 1, 'Electronic check': 2}
    input_data = pd.DataFrame([[age, tenure, monthly, total, products,
                                 internet_enc, contract_map[contract],
                                 payment_map[payment], calls]], columns=X.columns)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.error(f"⚠️ HIGH CHURN RISK! Probability: {probability:.1%}")
        st.warning("Actions: Offer discount, upgrade contract, assign account manager")
    else:
        st.success(f"✅ LOW CHURN RISK! Probability: {probability:.1%}")
        st.info("Customer is likely to stay. Consider upselling opportunities!")
