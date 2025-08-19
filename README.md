# FAISS RAG Chatbot

A lightweight console-based chatbot that uses **Cohere LLMs** with **FAISS vector search** for Retrieval-Augmented Generation (RAG).  
The system reads a local text file (`file.txt`), chunks it, embeds the chunks with **Cohere embeddings**, and stores them in a FAISS index for fast semantic search.  
User queries are embedded, the most relevant chunk(s) are retrieved, and then passed into Cohere's `command-r` chat model for contextual answers.

---

## Features
- Uses **FAISS** for in-memory similarity search
- **Cohere embeddings** (`embed-multilingual-v3.0`) for document and query vectors
- Document chunking with **LangChain’s RecursiveCharacterTextSplitter**
- Conversational memory with chat history
- Console-based interface (no API/DB required)
- Lightweight and easy to test

---

## Requirements
- Python 3.9+
- Cohere API key
