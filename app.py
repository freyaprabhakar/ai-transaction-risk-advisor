
import streamlit as st
import pandas as pd
import os
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="AI Risk Advisor", layout="wide")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# -----------------------------
# Knowledge base (RAG)
# -----------------------------
documents = [
    "High-value transactions above 200000 are often flagged for fraud risk.",
    "TRANSFER and CASH_OUT transactions are more vulnerable to fraud.",
    "Sudden spikes in transaction amount can indicate suspicious activity.",
    "Frequent transactions in a short time window may signal fraud.",
    "Unusual transaction behavior compared to normal patterns is risky."
]

# Embeddings + vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_texts(documents, embeddings)

# -----------------------------
# LLM setup
# -----------------------------
client = Groq(api_key=os.getenv("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"))

# -----------------------------
# UI
# -----------------------------
st.title("💳 AI Transaction Risk Advisor")
st.markdown("Simple AI-powered tool to assess transaction risk")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", min_value=0)

with col2:
    txn_type = st.selectbox(
        "Transaction Type",
        ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]
    )

analyze = st.button("Analyze")

# -----------------------------
# Analysis logic
# -----------------------------
if analyze:

    # Basic rule-based scoring
    risk_score = 0

    if amount > 200000:
        risk_score += 2
    elif amount > 50000:
        risk_score += 1

    if txn_type in ["TRANSFER", "CASH_OUT"]:
        risk_score += 1

    # Risk label
    if risk_score >= 3:
        risk_level = "High"
    elif risk_score == 2:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # -----------------------------
    # RAG retrieval
    # -----------------------------
    query = f"Transaction amount {amount} and type {txn_type}"

    docs = vectorstore.similarity_search(query, k=2)
    context = " ".join([doc.page_content for doc in docs])

    # -----------------------------
    # Prompt + LLM
    # -----------------------------
    prompt = f"""
    You are a financial analyst.

    Context:
    {context}

    Explain why this transaction might be risky.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content

    except:
        explanation = "Unable to generate explanation."

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("Result")

    st.write(f"**Risk Level:** {risk_level}")

    score = min(risk_score * 30, 100)
    st.progress(score)
    st.write(f"Risk Score: {score}/100")

    st.subheader("Explanation")
    st.write(explanation)
