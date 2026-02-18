import os
import streamlit as st
import pandas as pd
import yfinance as yfstre
import plotly.graph_objects as go
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import yfinance as yf


from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Financial & Research Copilot",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Financial & Research Copilot")

# ---------------------------------------------------
# API KEY INPUT
# ---------------------------------------------------

groq_api_key = st.text_input("🔑 Enter Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter Groq API key")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# ---------------------------------------------------
# MODEL (Free Tier Optimized)
# ---------------------------------------------------

model = Groq(
    id="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.2,
    max_tokens=300
)

# ---------------------------------------------------
# AGENTS
# ---------------------------------------------------

web_agent = Agent(
    name="web_agent",
    role="Research assistant",
    model=model,
    instructions="Be concise. Provide summarized insights."
)

finance_agent = Agent(
    name="finance_agent",
    role="Financial analysis assistant",
    model=model,
    instructions="Provide short financial insights."
)

# ---------------------------------------------------
# SESSION MEMORY
# ---------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🧠 Mode")
mode = st.sidebar.radio(
    "Select Mode",
    ["AI Research", "Finance Analysis", "Document Q&A (RAG)"]
)

# ---------------------------------------------------
# 1️⃣ AI RESEARCH
# ---------------------------------------------------

if mode == "AI Research":

    query = st.text_area("Enter research question")

    if st.button("Run Research") and query:

        with st.spinner("Thinking..."):

            response = web_agent.run(query)
            st.write(response.content)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": response.content}
            )

# ---------------------------------------------------
# 2️⃣ FINANCE MODE
# ---------------------------------------------------

elif mode == "Finance Analysis":

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)")
    period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "5y"])

    if st.button("Analyze Stock") and ticker:

        with st.spinner("Fetching Data..."):

            data = yf.download(ticker, period=period)

            if data.empty:
                st.error("Invalid ticker")
            else:

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode='lines',
                    name='Close Price'
                ))

                st.plotly_chart(fig, width='stretch')

                # AI Insight
                summary_prompt = f"""
                Provide short financial analysis for {ticker}
                Latest Close: {data['Close'].iloc[-1]}
                """

                response = finance_agent.run(summary_prompt)
                st.write(response.content)

# ---------------------------------------------------
# 3️⃣ RAG DOCUMENT Q&A
# ---------------------------------------------------

elif mode == "Document Q&A (RAG)":

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:

        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            text += page.extract_text()

        # Split into chunks
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

        # Embedding model
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embed_model.encode(chunks)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        query = st.text_input("Ask question from document")

        if st.button("Ask") and query:

            query_embedding = embed_model.encode([query])
            distances, indices = index.search(np.array(query_embedding), k=3)

            retrieved_chunks = [chunks[i] for i in indices[0]]
            context = "\n".join(retrieved_chunks)

            rag_prompt = f"""
            Answer using context below:
            {context}

            Question: {query}
            """

            response = web_agent.run(rag_prompt)
            st.write(response.content)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.caption("Built with Groq LLaMA 4 + Streamlit + FAISS + yFinance")
