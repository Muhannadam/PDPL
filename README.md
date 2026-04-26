# PDPL RAG Streamlit App

واجهة Streamlit لمساعد قانوني يعتمد على RAG لنظام حماية البيانات الشخصية السعودي.

## Important

This app does not rebuild the vector index.

It loads the existing FAISS files:

```text
data/vectorstore/faiss_index/index.faiss
data/vectorstore/faiss_index/index.pkl
