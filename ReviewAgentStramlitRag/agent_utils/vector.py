from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
import streamlit as st

# --- 1. Session State Initialization (ABSOLUTE TOP) ---
# This function ensures all necessary session state variables are set.
# It should be called first.
def initialize_session_state():
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        # st.write("DEBUG: Memory initialized for the first time.") # Debugging line
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # st.write("DEBUG: Messages initialized for the first time.") # Debugging line
        
# --- 3. Global Component Initialization (Cached) ---
# Use st.cache_resource for heavy objects that should persist across reruns
@st.cache_resource
def get_embedding_model():
    # st.write("DEBUG: Initializing OllamaEmbeddings...") # Debugging line
    return OllamaEmbeddings(model="mxbai-embed-large")

@st.cache_resource
def get_vector_store(_embedding_model_instance): # Pass the actual embedding model instance
    db_location = "./chrome_langchain_db_ui"
    collection_name = "restaurant_reviews_pdf"
    # st.write(f"DEBUG: Initializing Chroma DB at {db_location}...") # Debugging line
    return Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=_embedding_model_instance
    )

@st.cache_resource
def get_llm():
    # st.write("DEBUG: Initializing OllamaLLM (mistral)...") # Debugging line
    return OllamaLLM(model="mistral")
