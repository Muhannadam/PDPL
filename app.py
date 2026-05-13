import streamlit as st

from config import AppConfig
from rag_core import create_rag_chain, answer_question


st.set_page_config(
    page_title="PDPL Legal Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: "Segoe UI", Tahoma, Arial, sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .sub-title {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .metric-box {
        background-color: #f8f9fb;
        border: 1px solid #e6e8eb;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .source-box {
        direction: ltr;
        text-align: left;
        background-color: #f7f7f9;
        padding: 0.75rem;
        border-radius: 0.6rem;
        border: 1px solid #e5e5e5;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
    }

    .notice-box {
        background-color: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 0.75rem;
        padding: 0.9rem;
        color: #5d4037;
        margin-bottom: 1.2rem;
    }

    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_chain():
    config = AppConfig()
    return create_rag_chain(config)


with st.sidebar:
    st.title("PDPL Assistant")

    st.markdown("### حالة النظام")

    config = AppConfig()

    st.markdown(
        f"""
        <div class="metric-box">
        <b>Vectorstore</b><br>
        <span style="direction:ltr; display:block; text-align:left;">
        {config.vectorstore_path}
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-box">
        <b>Embedding Model</b><br>
        <span style="direction:ltr; display:block; text-align:left;">
        {config.embedding_model}
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-box">
        <b>Groq Model</b><br>
        <span style="direction:ltr; display:block; text-align:left;">
        {config.groq_model}
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_sources = st.toggle("إظهار المصادر", value=True)

    if st.button("مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.markdown(
    """
    <div class="main-title">مساعد قانوني لنظام حماية البيانات الشخصية السعودي</div>
    <div class="sub-title">
    يجيب النظام بناءً على الوثائق المفهرسة فقط، دون إعادة بناء الفهرس.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice-box">
    تنبيه: هذا المساعد لا يقدم استشارة قانونية ملزمة. راجع مختصًا قانونيًا قبل اتخاذ أي قرار.
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    chain = load_chain()
except Exception as exc:
    st.error("تعذر تحميل النظام. تأكد من وجود ملفات index.faiss و index.pkl ومن إضافة GROQ_API_KEY في Secrets.")
    st.exception(exc)
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and show_sources
            and message.get("sources")
        ):
            with st.expander("المصادر"):
                for source in message["sources"]:
                    st.markdown(
                        f'<div class="source-box">{source}</div>',
                        unsafe_allow_html=True,
                    )


query = st.chat_input("اكتب سؤالك هنا...")

if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في الوثائق..."):
            try:
                result = answer_question(query, chain=chain)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if show_sources and sources:
                    with st.expander("المصادر"):
                        for source in sources:
                            st.markdown(
                                f'<div class="source-box">{source}</div>',
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                error_message = "حدث خطأ أثناء معالجة السؤال."

                st.error(error_message)
                st.exception(exc)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
