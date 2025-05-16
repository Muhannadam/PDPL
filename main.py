import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# اجعل كل شيء يعمل على CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

st.set_page_config(page_title="مساعد حماية البيانات الشخصية", page_icon="🛡️")
st.title("مساعد نظام حماية البيانات الشخصية السعودي")
st.write("أدخل سؤالك حول النظام، وسيتم الرد بناءً على الوثائق الرسمية فقط.")

query = st.text_input("سؤالك هنا:", "")

if query:
    with st.spinner("جارٍ المعالجة..."):
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = HuggingFaceEmbeddings(model=model)
        db = FAISS.load_local("vectorstore", embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
            model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
        )
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = qa_chain.run(query)
        st.write("### الإجابة:")
        st.write(result)
