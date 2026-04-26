import os
import streamlit as st

from rag_engine import RAGEngine


st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #777;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}
.answer-box {
    padding: 1.2rem;
    border-radius: 14px;
    background: #f7f7f9;
    border: 1px solid #e5e5e5;
}
.source-box {
    padding: 0.9rem;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e8e8e8;
    margin-bottom: 0.7rem;
}
</style>
""", unsafe_allow_html=True)


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


@st.cache_resource
def load_engine(api_key):
    return RAGEngine(api_key)


st.markdown('<div class="main-title">🧠 RAG Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">اسأل ملفات Markdown الخاصة بك باستخدام FAISS + Groq + Streamlit</div>',
    unsafe_allow_html=True,
)

api_key = get_api_key()

if not api_key:
    st.error("لم يتم العثور على GROQ_API_KEY. أضفه في .streamlit/secrets.toml أو كمتغير بيئة.")
    st.stop()

engine = load_engine(api_key)

with st.sidebar:
    st.header("⚙️ الإعدادات")
    top_k = st.slider("عدد المقاطع المسترجعة", 1, 10, 5)
    show_sources = st.toggle("إظهار المقاطع المسترجعة", value=True)

    st.divider()
    st.caption("الموديل: llama-3.3-70b-versatile")
    st.caption("البحث: FAISS")
    st.caption("Embeddings: all-MiniLM-L6-v2")


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("اكتب سؤالك هنا...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("أبحث في الملفات وأجهز الإجابة..."):
            answer, chunks = engine.answer(query, top_k=top_k)

        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        if show_sources:
            with st.expander("📚 المقاطع المسترجعة"):
                for i, chunk in enumerate(chunks, start=1):
                    st.markdown(
                        f"""
<div class="source-box">
<b>المقطع {i}</b> — Score: {chunk['score']:.4f}<br><br>
{chunk['text'][:1200]}
</div>
""",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer})
