import streamlit as st
import requests

st.title("📚 Notebook LLM for Students")

# File upload
st.header("Upload Course Materials")
uploaded_files = st.file_uploader("Upload PDFs or TXT", accept_multiple_files=True)

if st.button("Ingest Materials"):
    if uploaded_files:
        files_to_upload = []
        for uploaded_file in uploaded_files:
            # tuple: (fieldname, (filename, fileobj, content_type))
            files_to_upload.append(
                ("files", (uploaded_file.name, uploaded_file, "application/octet-stream"))
            )

        response = requests.post("http://127.0.0.1:8000/upload/", files=files_to_upload)

        try:
            res_json = response.json()
            if "message" in res_json:
                st.success(res_json["message"])
            else:
                st.warning(f"No 'message' in response: {res_json}")
        except Exception as e:
            st.error(f"Failed to parse response: {e}\nRaw response: {response.text}")


    else:
        st.warning("Please upload files first.")



st.header("Ask Questions from Your Materials")
query = st.text_input("Type your question here:")

if st.button("Get Answer"):
    if query:
        try:
            # send JSON with key "query"
            response = requests.post(
                "http://127.0.0.1:8000/query/",
                json={"query": "what is whr"}   # <- make sure key matches backend
            )
            print(response.status_code, response.text)  # Debugging line
            res_json = response.json()
            if "answer" in res_json:
                st.success(res_json["answer"])
            else:
                st.warning(f"No 'answer' in response: {res_json}")
        except Exception as e:
            st.error(f"Failed to get answer: {e}")
    else:
        st.warning("Please enter a question first.")