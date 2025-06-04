from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

# Load restaurant review data from CSV
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Create embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Vector DB settings
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Optional: add documents if the DB folder doesn't exist
if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        doc = Document(
            page_content=f"{row['Title']} {row['Review']}",
            metadata={"rating": row["Rating"], "date": row["Date"]},
        )
        documents.append(doc)
        ids.append(str(i))

# Initialize or connect to Chroma DB
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents on first run
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

