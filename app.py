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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #FAFAF7;
    }

    /* Force readable text colors */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #1f2937;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0F4C3A 0%, #1a6b54 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(15, 76, 58, 0.15);
    }

    .app-header h1 {
        color: #ffffff !important;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
    }

    .app-header p {
        color: #d4e9e0 !important;
        margin: 0;
        font-size: 0.95rem;
    }

    /* Welcome state */
    .welcome-box {
        text-align: center;
        padding: 2.5rem 1rem;
    }

    .welcome-box h3 {
        color: #0F4C3A !important;
        margin-bottom: 0.5rem;
    }

    .welcome-box p {
        color: #6b7280 !important;
    }

    /* Source cards */
    .source-box {
        background: #ffffff;
        border-left: 3px solid #C9A961;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        line-height: 1.7;
        color: #1f2937 !important;
    }

    .source-box b {
        color: #0F4C3A !important;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.8rem;
        color: #6b7280 !important;
        text-align: center;
        padding: 0.8rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #0F4C3A !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton button {
        background-color: #f9fafb;
        color: #1f2937 !important;
        border: 1px solid #e5e7eb;
        text-align: left;
        font-weight: 500;
        transition: all 0.2s;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #0F4C3A;
        color: #ffffff !important;
        border-color: #0F4C3A;
    }

    /* Chat input */
    .stChatInput textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        color: #1f2937 !important;
    }

    /* Chat messages */
    .stChatMessage {
        background: transparent;
    }

    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage span {
        color: #1f2937 !important;
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
    st.markdown("### PDPL Assistant")
    st.markdown(
        "<p style='color:#6b7280; font-size:0.9rem;'>Intelligent Assistant specialized in the Saudi Personal Data Protection Law (PDPL).</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### Suggested Questions")
    suggested_questions = [
        "What are the rights of a data subject?",
        "When must a data breach be reported?",
        "What are the penalties for violating the regulation?",
        "What is meant by sensitive data?",
    ]

    for q in suggested_questions:
        if st.button(q, use_container_width=True, key=f"sq_{q}"):
            st.session_state.pending_query = q
            st.rerun()

    st.divider()

    show_sources = st.toggle("Show sources", value=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Technical information"):
        config = AppConfig()
        st.caption(f"**Embedding:** `{config.embedding_model}`")
        st.caption(f"**LLM:** `{config.groq_model}`")
        st.caption(f"**Index:** `{config.vectorstore_path}`")



try:
    chain = load_chain()
except Exception as exc:
    st.error("System load failed. Ensure that the 'index.faiss' and 'index.pkl' files exist, and that the 'GROQ_API_KEY' has been added to Secrets.")
    st.exception(exc)
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        """
        <div style='text-align:center; padding:2rem 1rem; color:#6b7280;'>
            <div style='font-size:3rem; margin-bottom:0.5rem;'></div>
            <h3 style='color:#0F4C3A; margin-bottom:0.5rem;'>How can I assist you today?</h3>
            <p>Ask your question regarding the Personal Data Protection Law, or select a question from the sidebar.</p>
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
            with st.expander("Sources"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(
                        f'<div class="source-box"><b>Source {idx + 1}:</b><br>{source}</div>',
                        unsafe_allow_html=True,
                    )


query = st.chat_input("Enter your question here...")

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
        with st.spinner("Searching documents..."):
            try:
                result = answer_question(query, chain=chain)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if show_sources and sources:
                    with st.expander("Sources"):
                        for idx, source in enumerate(sources):
                            st.markdown(
                                f'<div class="source-box"><b>Source {idx + 1}:</b><br>{source}</div>',
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
                error_message = "An error occurred while processing the question."

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
        This assistant does not provide binding legal advice. Please consult a legal professional before making any decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
