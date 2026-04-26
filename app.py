import streamlit as st

from config import AppConfig
from rag_core import create_rag_chain, answer_question


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="PDPL Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: "Segoe UI", Tahoma, Arial, sans-serif;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .sub-title {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
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

    .warning-box {
        background-color: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 0.6rem;
        padding: 0.8rem;
        margin-top: 1rem;
        color: #5d4037;
    }

    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Cached chain
# =========================================================
@st.cache_resource(show_spinner=False)
def get_chain():
    config = AppConfig()
    return create_rag_chain(config)


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.title("⚖️ PDPL Assistant")

    st.markdown("### الإعدادات")
    st.write("النظام يستخدم FAISS index المحفوظ مسبقًا، ولا يعيد بناء الفهرس.")

    config = AppConfig()

    st.code(
        f"""Embedding model:
{config.embedding_model}

Groq model:
{config.groq_model}

retrieval_k:
{config.retrieval_k}

final_max_sources:
{config.final_max_sources}
""",
        language="text",
    )

    show_sources = st.toggle("إظهار المصادر", value=True)
    show_disclaimer = st.toggle("إظهار تنبيه الاستخدام", value=True)

    if st.button("مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <div class="main-title">مساعد قانوني لنظام حماية البيانات الشخصية السعودي</div>
    <div class="sub-title">
    اسأل عن محتوى الوثائق المرفوعة، وسيجيب النظام فقط بناءً على السياق المسترجع من الفهرس.
    </div>
    """,
    unsafe_allow_html=True,
)

if show_disclaimer:
    st.markdown(
        """
        <div class="warning-box">
        هذا النظام مساعد بحثي وليس بديلاً عن الاستشارة القانونية المهنية.
        النتائج تعتمد على الوثائق المفهرسة فقط.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Session state
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# Load chain
# =========================================================
try:
    chain = get_chain()
except Exception as exc:
    st.error("تعذر تحميل النظام.")
    st.exception(exc)
    st.stop()


# =========================================================
# Render history
# =========================================================
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


# =========================================================
# Input
# =========================================================
user_query = st.chat_input("اكتب سؤالك هنا...")

if user_query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في الوثائق..."):
            try:
                result = answer_question(user_query, chain=chain)
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
                error_msg = "حدث خطأ أثناء معالجة السؤال."
                st.error(error_msg)
                st.exception(exc)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                    }
                )
