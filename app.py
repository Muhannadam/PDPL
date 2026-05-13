import streamlit as st

from config import AppConfig
from rag_core import create_rag_chain, answer_question


st.set_page_config(
    page_title="PDPL Legal Assistant",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
    }

    .stApp {
        background-color: #FAFAF7;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0F4C3A 0%, #1a6b54 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 12px rgba(15, 76, 58, 0.15);
    }

    .app-header h1 {
        color: white;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
    }

    .app-header p {
        color: #d4e9e0;
        margin: 0;
        font-size: 0.95rem;
    }

    /* Suggested question cards */
    .suggestion-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.6rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    .suggestion-card:hover {
        border-color: #0F4C3A;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Source cards */
    .source-box {
        background: white;
        border-right: 3px solid #C9A961;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        line-height: 1.7;
    }

    /* Disclaimer — smaller and softer */
    .disclaimer {
        font-size: 0.8rem;
        color: #6b7280;
        text-align: center;
        padding: 0.6rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-left: 1px solid #e5e7eb;
    }

    /* Chat input */
    .stChatInput textarea {
        font-family: 'Tajawal', sans-serif !important;
        font-size: 1rem !important;
    }

    .stChatMessage {
        direction: rtl;
        text-align: right;
        background: transparent;
    }

    /* Hide Streamlit default footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
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
    <div class="app-header">
        <h1>⚖️ مساعد نظام حماية البيانات الشخصية</h1>
        <p>إجابات مدعومة بالوثائق الرسمية لنظام PDPL السعودي</p>
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

if not st.session_state.messages:
    st.markdown(
        """
        <div style='text-align:center; padding:2rem 1rem; color:#6b7280;'>
            <div style='font-size:3rem; margin-bottom:0.5rem;'>📜</div>
            <h3 style='color:#0F4C3A; margin-bottom:0.5rem;'>كيف يمكنني مساعدتك اليوم؟</h3>
            <p>اطرح سؤالك حول نظام حماية البيانات الشخصية، أو اختر سؤالًا من القائمة الجانبية.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and show_sources
            and message.get("sources")
        ):
            with st.expander("المصادر"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(
                        f'<div class="source-box"><b>📄 مصدر {idx + 1}:</b><br>{source}</div>',
                        unsafe_allow_html=True,
                    )


query = st.chat_input("اكتب سؤالك هنا...")

# Handle suggested question click from sidebar
if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query
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
                                f'<div class="source-box"><b>📄 مصدر {idx + 1}:</b><br>{source}</div>',
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

st.markdown(
    """
    <div class="disclaimer">
        ⚠️ هذا المساعد لا يقدم استشارة قانونية ملزمة. يُرجى مراجعة مختص قانوني قبل اتخاذ أي قرار.
    </div>
    """,
    unsafe_allow_html=True,
)
