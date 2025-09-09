"""
Main entry point for the Aura AI Research Assistant Streamlit app.
Handles PDF upload, document processing, hybrid search (BM25 + dense vectors),
and interacts with a local Mistral-7B language model to answer credit risk and financial research questions.
"""

# ====== Imports ======
import os
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
import re
from config import Config
from utils import get_pdf_text, get_text_chunks, highlight_relevant_text
from vector_store import VectorStore
import hybrid_rerank
from mistral_client import ask_mistral
nest_asyncio.apply()

def main():
    load_dotenv()

    # ====== Streamlit UI Setup ======
    st.set_page_config(
        page_title="Aura AI Research Assistant",
        page_icon="📊",
        layout="centered"
    )

    st.markdown("""
    <div style="text-align:center; margin-bottom:30px">
        <h1 style="display:inline-block; border-bottom:2px solid #4f8bf9; padding-bottom:10px">
        📊 Aura AI Credit Risk Research Assistant
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # ====== Initialize session state ======
    for key in ["vector_store", "corpus", "chat_history"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "chat_history" else []

    # ====== Sidebar - Chat History ======
    with st.sidebar:
        st.subheader("Chat History")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
        st.divider()
        for idx, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{idx+1}: {q[:50]}..." if len(q) > 50 else f"Q{idx+1}: {q}"):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")
        st.divider()

    # ====== Upload Financial Documents ======
    with st.container():
        st.subheader("📂 Upload Financial Reports and Credit Documents")
        pdf_docs = st.file_uploader(
            "Drag and drop PDF files related to credit risk and financial analysis here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        # ====== Build Knowledge Base Button ======
        if st.button("⚙️ Build Research Knowledge Base", type="primary", disabled=not pdf_docs):
            with st.spinner("Processing uploaded documents..."):
                # Extract text from PDFs
                raw_docs = get_pdf_text(pdf_docs)
                st.info(f"Extracted {len(raw_docs)} pages from uploaded documents.")

                # Break text into chunks suitable for vectorization
                text_chunks = get_text_chunks(raw_docs)
                vector_store = VectorStore()
                vector_store.ensure_collection()

                # Progress bar for chunk processing
                progress_bar = st.progress(0, text="Processing document chunks...")

                def update_progress(percent):
                    progress_bar.progress(percent, text=f"Processing... {percent}%")

                # Add chunks to vector store with progress updates
                corpus = vector_store.add_documents(text_chunks, progress_callback=update_progress)
                st.session_state.vector_store = vector_store
                st.session_state.corpus = corpus
                
                st.success(f"✅ Loaded {len(text_chunks)} chunks from {len(pdf_docs)} document(s)!")
                progress_bar.empty()

    # ====== User Query Input ======
    user_question = st.text_input(
        "Ask your AI researcher about credit risk and financial insights:",
        placeholder="Type your research question here...",
        label_visibility="collapsed"
    )

    # ====== Generate Answer from Mistral LLM ======
    if user_question and st.session_state.vector_store:
        corpus = st.session_state.vector_store.get_corpus()
        if not corpus:
            st.warning("⚠️ No documents available. Please upload and process PDFs first.")
        else:
            try:
                results = hybrid_rerank.hybrid_search(
                    user_question,
                    st.session_state.vector_store.client,
                    st.session_state.vector_store.collection_name,
                    corpus,
                    limit=8
                )
            except ValueError as e:
                st.error(str(e))
                results = []

            if results:
                context_chunks = []
                seen_sections = set()
                for hits in results:
                    if isinstance(hits, (list, tuple)):
                        for hit in hits:
                            entity = hit.get("entity", hit)
                            metadata = entity.get("metadata", {})
                            text = entity.get("text", "")
                            section = metadata.get("section", "").lower()
                            if section in {"executive summary", "overview"} and section not in seen_sections:
                                context_chunks.insert(0, text)
                                seen_sections.add(section)
                            else:
                                context_chunks.append(text)

                full_context = "\n\n".join(context_chunks)

                # Debug mode to view retrieved context
                debug = st.checkbox("🐞 Show retrieved context (debug mode)")
                if debug:
                    with st.expander("Retrieved Context"):
                        st.markdown(full_context)

                # Construct system prompt tailored for credit risk research
                prompt = (
                    "You are Aura AI, a precise and domain-expert credit risk research assistant. "
                    "Answer using ONLY the information in the given context. "
                    "Do NOT guess or go beyond the context unless explicitly asked. "
                    "Avoid repeated answers; phrase responses carefully. "
                    "Present clear, well-structured answers referencing dataset specifics, methodologies, or evaluation metrics. "
                    "If info is missing, state: 'This topic is not mentioned in the provided context.' "
                    f"Context:\n{full_context}\n\n"
                    f"Question: {user_question}\n"
                    "Answer:"
                )

                answer = ask_mistral(prompt, max_tokens=768, temperature=0.2)

                st.session_state.chat_history.append((user_question, answer))

                # Display Answer
                st.markdown("---")
                st.subheader("💡 AI Research Assistant Answer")
                st.markdown(answer)
                st.divider()

                # Show source context references
                with st.expander("🔍 Show Source Context"):
                    flat_hits = []
                    for hits in results:
                        if isinstance(hits, (list, tuple)):
                            for hit in hits:
                                entity = hit.get("entity", hit)
                                metadata = entity.get("metadata", {})
                                text = entity.get("text", "")
                                flat_hits.append((metadata, text))

                    for idx, (metadata, text) in enumerate(flat_hits[:3]):
                        source = metadata.get("source", "unknown")
                        page = metadata.get("page", "N/A")
                        line = metadata.get("line", "N/A")
                        with st.container():
                            st.markdown(f"**📄 Source {idx+1}:** `{source}` (Page {page}, Line {line})")
                            st.caption(highlight_relevant_text(text, user_question))
                            st.divider()


if __name__ == "__main__":
    main()
