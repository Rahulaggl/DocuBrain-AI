import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and Model Configurations
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Groq API key
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! How can I assist you?"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]

# Introduction Section
def display_introduction():
    st.title("Welcome to DocuBrain AI Assistant")
    st.write("""
    **DocuBrain** offers the following functionalities:
    1. **Text Processing**: Summarize articles, extract highlights, and generate Points of Minutes (PoM).
    2. **InsightDoc AI Analyzer**: Upload PDFs for document analysis and search.
    3. **Chat with Assistant**: Ask questions and receive tailored responses.
    """)
    st.write("### Download Sample Files:")
    st.write("Visit [GitHub Repo](https://github.com/Rahulaggl/DocuBrain-AI) for sample files.")

# Text Processing Section
def process_text_with_groq(task, text):
    try:
        task_map = {
            "summary": "Summarize the following text.",
            "highlight": "Extract the highlights from the text.",
            "point of minutes": "Generate a concise point of minutes (PoM) for the following text.",
            "custom": "Follow the user's custom instructions for the given text."
        }
        prompt = f"{task_map[task]}: {text}"
        # Replace with actual Groq API call if available
        return f"[Simulated Response] {prompt}"
    except Exception as e:
        return f"Error: {e}"

def display_text_processing():
    st.subheader("Text Processing")
    task = st.selectbox("Select Task", ["summary", "highlight", "point of minutes", "custom"], index=0)
    user_input = st.text_area("Enter your text:", height=200)
    if st.button("Process") and user_input:
        with st.spinner("Processing..."):
            output = process_text_with_groq(task, user_input)
            st.success(output)

# Document Search Section
def create_conversational_chain(vector_store):
    llm = HuggingFaceHub(
        repo_id=MODEL_NAME,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        model_kwargs={"temperature": 0.75, "top_p": 0.9, "max_length": 1024}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

def display_document_search(chain):
    st.subheader("Document Search")
    query = st.text_input("Ask a question:")
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.write("### Answer:")
            st.success(result["answer"])
            if result.get("source_documents"):
                st.write("### Source Context:")
                for doc in result["source_documents"]:
                    st.write(doc.page_content)

# Chat Section
def chat_with_groq(message_text):
    try:
        # Replace with Groq API call
        return f"[Simulated Response] {message_text}"
    except Exception as e:
        return f"Error: {e}"

def display_chat_window():
    st.subheader("Chat with Assistant")
    user_input = st.text_input("Your message:")
    if st.button("Send") and user_input:
        with st.spinner("Thinking..."):
            response = chat_with_groq(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
                message(st.session_state["generated"][i], key=f"{i}")

# Main Application
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a mode",
        ["Introduction", "Text Processing", "Document Search", "Chat Window"]
    )
    initialize_session_state()
    if app_mode == "Introduction":
        display_introduction()
    elif app_mode == "Text Processing":
        display_text_processing()
    elif app_mode == "Document Search":
        uploaded_files = st.sidebar.file_uploader("Upload PDFs:", accept_multiple_files=True)
        if uploaded_files:
            docs = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    loader = PyPDFLoader(temp_file.name)
                    docs.extend(loader.load())
                    os.remove(temp_file.name)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
            chunks = text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embedding=embeddings)
            chain = create_conversational_chain(vector_store)
            display_document_search(chain)
    elif app_mode == "Chat Window":
        display_chat_window()

if __name__ == "__main__":
    main()
