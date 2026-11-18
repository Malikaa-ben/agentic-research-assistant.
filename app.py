"""Interface Streamlit : streamlit run app.py"""
import streamlit as st

from src.agent import build_agent, run_agent
from src.config import PDF_DIR
from src.rag import build_index

st.set_page_config(page_title="Agentic AI Research Assistant", page_icon="🤖")
st.title(" Agentic AI Research Assistant")
st.caption("LangGraph · Llama 3.2 (Ollama) · FAISS")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Ajouter des PDF", type="pdf", accept_multiple_files=True)
    if uploaded:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (PDF_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"{len(uploaded)} fichier(s) enregistré(s)")
    if st.button("Indexer les PDF"):
        with st.spinner("Indexation en cours..."):
            try:
                n_pdf, n_chunks = build_index()
                st.success(f"{n_pdf} PDF, {n_chunks} passages indexés")
            except Exception as e:
                st.error(str(e))

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for s in msg.get("steps", []):
            with st.expander(f" {s['tool']}({s['args']})"):
                st.code(s["output"])

if question := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
            out = run_agent(st.session_state.agent, question, history)
        st.markdown(out["answer"])
        for s in out["steps"]:
            with st.expander(f" {s['tool']}({s['args']})"):
                st.code(s["output"])
    st.session_state.messages.append(
        {"role": "assistant", "content": out["answer"], "steps": out["steps"]}
    )
