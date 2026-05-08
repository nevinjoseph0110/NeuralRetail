import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NeuralRetail Dashboard",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🛒 NeuralRetail Dashboard")
st.write("AI-Powered Retail Analytics System")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/cleaned_retail.csv")

segments = pd.read_csv("data/customer_segments.csv")

# -----------------------------
# KPI SECTION
# -----------------------------
total_sales = df['TotalPrice'].sum()

total_orders = df['InvoiceNo'].nunique()

total_customers = df['CustomerID'].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Sales", f"${total_sales:,.2f}")

col2.metric("📦 Total Orders", total_orders)

col3.metric("👥 Customers", total_customers)

# -----------------------------
# SALES TREND
# -----------------------------
st.subheader("📈 Daily Sales Trend")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum()

st.line_chart(daily_sales)

# -----------------------------
# TOP PRODUCTS
# -----------------------------
st.subheader("🛒 Top Products")

top_products = df.groupby('Description')['Quantity'].sum() \
                 .sort_values(ascending=False) \
                 .head(10)

st.bar_chart(top_products)

# -----------------------------
# CUSTOMER SEGMENTS
# -----------------------------
st.subheader("👥 Customer Segments")

segment_counts = segments['Cluster'].value_counts()

st.bar_chart(segment_counts)

# -----------------------------
# CHURN PREDICTION
# -----------------------------
st.subheader("⚠️ Churn Prediction")

# Load saved model
with open("models/churn_model.pkl", "rb") as file:
    model = pickle.load(file)

recency = st.number_input("Recency")
frequency = st.number_input("Frequency")
monetary = st.number_input("Monetary")

if st.button("Predict Churn"):

    sample = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    }])

    prediction = model.predict(sample)[0]

    if prediction == 1:
        st.error("⚠️ High Churn Risk")
    else:
        st.success("✅ Low Churn Risk") 