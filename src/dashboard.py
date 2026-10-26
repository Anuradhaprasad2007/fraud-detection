import requests
import streamlit as st
import pandas as pd

API_URL = "http://127.0.0.1:8000"  # change this one line when backend is deployed

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Fraud Detection in Online Payments")

tab1, tab2 = st.tabs(["Check a Transaction", "Recent Transactions"])

with tab1:
    st.subheader("Enter transaction details")

    col1, col2 = st.columns(2)
    with col1:
        step = st.number_input("Step (hour of simulation)", min_value=1, value=1)
        txn_type = st.selectbox("Transaction type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
        amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    with col2:
        oldbalanceOrg = st.number_input("Sender balance before", min_value=0.0, value=1000.0)
        newbalanceOrig = st.number_input("Sender balance after", min_value=0.0, value=0.0)
        oldbalanceDest = st.number_input("Receiver balance before", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("Receiver balance after", min_value=0.0, value=0.0)

    if st.button("Check for Fraud", type="primary"):
        payload = {
            "step": step,
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()

            if result.get("is_fraud"):
                st.error(f"FRAUD DETECTED — probability: {result['fraud_probability']:.2%}")
            else:
                st.success(f"Looks legitimate — fraud probability: {result['fraud_probability']:.2%}")

            st.write("**Top factors behind this decision:**")
            factors_df = pd.DataFrame(result["top_factors"])
            st.table(factors_df)

        except Exception as e:
            st.error(f"Could not reach the backend: {e}")

with tab2:
    st.subheader("Recent predictions")
    try:
        response = requests.get(f"{API_URL}/transactions?limit=50")
        records = response.json()
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No transactions recorded yet.")
    except Exception as e:
        st.error(f"Could not reach the backend: {e}")