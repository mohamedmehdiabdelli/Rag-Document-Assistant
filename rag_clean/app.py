import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("📄 AI Document Assistant (RAG)")

# Initialize pipeline (stored in session so it doesn't reload every time)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "loaded_filename" not in st.session_state:
    st.session_state.loaded_filename = None

if "loaded" not in st.session_state:
    st.session_state.loaded = False


# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and uploaded_file.name != st.session_state.loaded_filename:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.pipeline.load_document("temp.pdf")
    st.session_state.loaded = True
    st.session_state.loaded_filename = uploaded_file.name

    # Clear any previous answer since the document changed
    st.session_state.pop("last_result", None)

    st.success("Document loaded successfully!")


# ----------------------------
# QUESTION INPUT
# ----------------------------
question = st.text_input("Ask a question about the document")

if st.button("Ask") and st.session_state.loaded and question:
    # Store the result in session_state so it survives reruns
    # (e.g. when the "Show retrieval details" checkbox is toggled below)
    st.session_state.last_result = st.session_state.pipeline.ask(question)


# ----------------------------
# ANSWER + DEBUG (outside the button block so it persists across reruns)
# ----------------------------
if "last_result" in st.session_state:

    result = st.session_state.last_result

    st.subheader("Answer")
    st.write(result["answer"])

    show_debug = st.checkbox("🔍 Show retrieval details")

    if show_debug:

        st.subheader("Retrieved Chunks")

        for i, r in enumerate(result["retrieved_chunks"]):

            st.markdown(f"### Chunk {i+1}")
            st.write("📄 Document:", r["document"])
            st.write("📍 Page:", r["page"])
            if r["distance"] is not None:
                st.write("📊 Similarity:", round(r["distance"], 4))
            else:
                st.write("📊 Similarity: Neighbor chunk")
            st.write(r["text"])
            st.divider()