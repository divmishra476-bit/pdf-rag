import streamlit as st
from ingest import ingest_pdf  # (num_chunks = ingest_pdf("test_doc.pdf")) stays silent since __name__ won't be "__main__" in this context
from query import ask_question  # same idea, borrowing ask_question from query.py

st.title("Document Q&A Assistant")

# st.session_state persists across reruns (Streamlit reruns the whole script on every click/input) -
# a normal variable would reset to [] every single time, so this is the only way to accumulate history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())  # writes the actual uploaded file's bytes into that new file

    # this must be OUTSIDE the "with open" block (same indent as it, not nested inside) -
    # writing the file and clicking "process" are two separate, independent actions
    if st.button("Process document"):
        with st.spinner("Processing..."):
            num_chunks = ingest_pdf("uploaded.pdf")
        st.success(f"Document processed into {num_chunks} chunks. Ask away!")

question = st.text_input("Ask a question about the document")

if question:
    with st.spinner("Thinking..."):
        answer = ask_question(question)
    st.session_state.chat_history.append((question, answer))  # (question, answer) is a tuple - a pair bundled together, added to the running history

# loop through every stored pair and display it, so past Q&As stay visible instead of being overwritten
for q, a in st.session_state.chat_history:
    st.write(f"**Q:** {q}")
    st.write(f"**A:** {a}")
    st.write("---")