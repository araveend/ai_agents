import streamlit as st
import os
from pypdf import PdfReader
from PIL import Image

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from agent_utils.vector import  get_embedding_model,get_vector_store,get_llm,initialize_session_state
from agent_utils.template import template

# Call the initialization function at the very beginning
initialize_session_state()

# Always refer to it like this:
memory = st.session_state.memory

#st.write("Memory initialized:", "memory" in st.session_state)



# --- 2. UI Header ---
# Open and resize image
image = Image.open("images/pizza.jpg")
resized_image = image.resize((800, 200))  # Width x Height in pixels

# Display resized image
st.image(resized_image)
st.title(" Pizza Review Assistant with Memory")
st.markdown("Upload a **PDF with reviews**, then ask questions. Chat history is remembered.")



# Instantiate cached components
embedding_model = get_embedding_model()
vector_store = get_vector_store(embedding_model) # Pass the instance
llm = get_llm()



prompt = ChatPromptTemplate.from_template(template)


# --- 5. Build LCEL Chain ---
# This part is crucial for accessing chat_history from session_state.memory
chain = (
    RunnableMap({
        "question": lambda x: x["question"],
        "reviews": lambda x: x["reviews"],
        # Correctly load chat history from the initialized session_state.memory
        "chat_history": lambda x: memory.load_memory_variables({})["chat_history"],
    })
    | prompt
    | llm
    | StrOutputParser()
)


# --- 6. PDF Upload Section ---
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Processing PDF and updating knowledge base..."):
        pdf_reader = PdfReader(uploaded_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

        # Increase chunk size for better context, adjust as needed
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata={"source": uploaded_file.name}) for chunk in chunks]

        # Add documents to vector store
        vector_store.add_documents(documents)
        st.success(f"{len(documents)} chunks embedded and added to Chroma DB from **{uploaded_file.name}**.")
    # st.rerun() # Optional: Rerun immediately after upload to clear file uploader and refresh UI


# --- 7. Chat Interface ---
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your question about the pizza restaurant..."):
    # Add user message to chat history for display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get reviews from retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # Use the current question as the query for the retriever
    docs = retriever.invoke(prompt)
    reviews = "\n".join([doc.page_content for doc in docs])

    # Run chain and get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke({
                    "question": prompt, # Pass the original prompt as "question"
                    "reviews": reviews
                })
                st.markdown(response)

                # Save current interaction to memory
                memory.save_context(
                    {"input": prompt},
                    {"output": response}
                )
                # Add assistant response to chat history for display
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Optional: Show retrieved documents for debugging
                with st.expander("🔍 Retrieved Documents"):
                    if docs:
                        for i, doc in enumerate(docs):
                            st.write(f"**Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}):**")
                            st.write(doc.page_content)
                            st.write("---")
                    else:
                        st.info("No relevant documents retrieved.")

                # Optional: Show raw chat history
                with st.expander("🧠 Raw Chat History (for debugging)"):
                    if st.session_state.memory.chat_memory.messages:
                        st.json([m.dict() for m in st.session_state.memory.chat_memory.messages])
                    else:
                        st.info("Chat history is empty.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e) # This will show a full traceback in the Streamlit app
                # Also print to terminal for easier debugging
                print(f"\n!!! ERROR during chain invocation: {e} !!!")
                import traceback
                traceback.print_exc()