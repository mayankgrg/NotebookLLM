import streamlit as st
import requests

st.title("📚 Notebook LLM for Students")

# File upload
st.header("Upload Course Materials")
uploaded_files = st.file_uploader("Upload PDFs or TXT", accept_multiple_files=True)
if st.button("Ingest Materials"):
    if uploaded_files:
        files_to_upload = {"files": [(f.name, f, "application/octet-stream") for f in uploaded_files]}
        response = requests.post("http://127.0.0.1:8000/upload/", files=files_to_upload)
        st.success(response.json()["message"])
    else:
        st.warning("Please upload files first.")

# Query section
st.header("Ask Questions / Generate Quiz / Flashcards")
user_query = st.text_input("Enter your query here...")
if st.button("Get Answer"):
    if user_query:
        response = requests.post("http://127.0.0.1:8000/query/", json={"query": user_query})
        st.write("**Answer:**")
        st.write(response.json()["answer"])
