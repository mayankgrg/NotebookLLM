# 📚 Notebook LLM - AI Study Assistant

Notebook LLM is an end-to-end AI-powered study assistant that allows students to upload course materials and generate answers, quizzes, flashcards, and summaries using **RAG (Retrieval-Augmented Generation)** and **Large Language Models (LLMs)**. It features a **Streamlit frontend** for easy interaction and a **FastAPI backend** for processing and knowledge retrieval.

---

## Features

- Upload course materials (PDF, TXT, DOCX)
- Ingest documents into a **vector database**
- Ask questions using a **chatbot interface**
- Generate **quizzes, flashcards, and summaries**
- Powered by **RAG + LLMs** for accurate context-aware answers

---

## Tech Stack

- **Frontend:** Streamlit  
- **Backend:** FastAPI + Uvicorn  
- **Embedding Model:** SentenceTransformers (`all-MiniLM-L6-v2`)  
- **Vector Database:** FAISS  
- **LLM:** OpenAI GPT-4 (or local LLM of choice)  
- **File Parsing:** PyPDF2 (PDF), python-docx (optional DOCX)  

---

## Project Structure

