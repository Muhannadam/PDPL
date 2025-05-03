
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import os

st.set_page_config(page_title="مساعد حماية البيانات الشخصية", page_icon="🛡️")

st.title("مساعد نظام حماية البيانات الشخصية السعودي")
st.write("أدخل سؤالك حول النظام، وسيتم الرد بناءً على الوثائق الرسمية فقط.")

query = st.text_input("سؤالك هنا:", "")

if query:
    with st.spinner("جارٍ المعالجة..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        db = FAISS.load_local("vectorstore", embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature":0.5, "max_new_tokens":512})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = qa_chain.run(query)
        st.write("### الإجابة:")
        st.write(result)
