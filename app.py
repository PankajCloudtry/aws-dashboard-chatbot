import streamlit as st
from PIL import Image
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

logo = Image.open("logo.png")
st.set_page_config(page_title="AWS PDF Analyzer", layout="wide", page_icon=logo)
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=180)
with col2:
    st.title("📊 AWS PDF Billing Analyzer")

uploaded_file = st.file_uploader("📎 Upload AWS Billing PDF", type="pdf")

if uploaded_file:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp_uploaded.pdf")
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])

    service_keywords = [
        "Elastic Compute Cloud", "EC2", "Simple Storage Service", "S3",
        "Virtual Private Cloud", "VPC", "EBS", "RDS", "Lambda", "CloudWatch"
    ]

    service_costs = []
    for line in full_text.splitlines():
        for service in service_keywords:
            if service in line:
                match = re.search(rf"{re.escape(service)}.*?USD\s*([0-9,]+\.\d{{2}})", line)
                if match:
                    amount = float(match.group(1).replace(",", ""))
                    service_costs.append({"Service": service, "Cost": amount})
                break

    if not service_costs:
        st.error("❌ No valid AWS service cost entries found.")
    else:
        df = pd.DataFrame(service_costs).groupby("Service", as_index=False).sum()
        df = df.sort_values(by="Cost", ascending=False)

        st.subheader("💰 Top 3 Costly Services")
        st.table(df.head(3))

        fig, ax = plt.subplots()
        top_services = df.head(5)
        ax.pie(top_services["Cost"], labels=top_services["Service"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(pages, embeddings)
        db.save_local("pdf_vector_store")

        st.subheader("💬 Ask Questions from Your PDF")
        question = st.text_input("Type your question:")
        if question:
            chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                retriever=db.as_retriever(),
                return_source_documents=False
            )
            answer = chain.run(question)
            st.success(answer)
else:
    st.info("📄 Please upload a PDF to get started.")
