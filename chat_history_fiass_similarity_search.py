import cohere
import os
from dotenv import load_dotenv
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load API key
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Load static context
with open("file.txt", "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

# Wrap the raw text into a Document for chunking
documents = [Document(page_content=raw_text)]

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# Extract text chunks
docs = [doc.page_content for doc in chunks]

# Embed chunks using correct input_type
embed_resp = co.embed(
    texts=docs,
    model="embed-multilingual-v3.0",
    input_type="search_document"  # REQUIRED for document vectors
)

# Convert to NumPy and normalize
vectors = np.array(embed_resp.embeddings, dtype="float32")
faiss.normalize_L2(vectors)

# Create FAISS index
dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(vectors)

# Chat history
chat_history = []
system_msg = "You are a helpful assistant."

# Chat loop
print("Start chatting (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: successfully exit!")
        break
    if not user_input.strip():
        print("Bot: Please type something.")
        continue

    # Embed the user input correctly
    qresp = co.embed(
        texts=[user_input],
        model="embed-multilingual-v3.0",
        input_type="search_query"  # REQUIRED for queries
    )

    #qvec is a vector representation of the user’s question.

    qvec = np.array(qresp.embeddings, dtype="float32")
    faiss.normalize_L2(qvec)

    # Retrieve top-1 chunk
    _, I = index.search(qvec, 1)
    best_chunk = docs[I[0][0]]

    # Pass top chunk to co.chat
    response = co.chat(
        model="command-r",
        message=user_input,
        temperature=0.5,
        documents=[{"text": best_chunk}],
        chat_history=chat_history,
        preamble=system_msg
        #top_k=1
        #k=3
        #p=0.77
    )

    bot_reply = response.text.strip()
    print("Bot:", bot_reply)

    # Save history
    chat_history.append({"role": "USER", "message": user_input})
    chat_history.append({"role": "CHATBOT", "message": bot_reply})
