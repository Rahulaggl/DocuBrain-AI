import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from groq import Groq
import os
import tempfile
from dotenv import load_dotenv
import tmol  # For secure API key management

# Load environment variables from tmol
HUGGINGFACE_API_KEY = tmol.get("HUGGINGFACE_API_KEY")  # Access Hugging Face API key
GROQ_API_KEY = tmol.get("GROQ_API_KEY")  # Access Groq API key

# Model Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize Groq client with the API key
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! How can I assist you?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]

# Part 0: Introduction
def display_introduction():
    st.title("Welcome to DocuBrain AI Assistant")
    st.write("""
    **DocuBrain** is a powerful tool offering functionalities:
    
    1. **Text Processing**: Summarize articles, extract highlights, generate Points of Minutes (PoM), and follow custom instructions.
    2. **InsightDoc AI Analyzer**: Upload PDFs for comparison, summarization, and search-based tasks.
    3. **Chat with Assistant**: Ask questions and get personalized responses.
    """)

# Part 1: Summary, Highlights, PoM, and Custom Instructions
def process_text_with_groq(task, text):
    try:
        task_map = {
            "summary": "Summarize the following text.",
            "highlight": "Extract the highlights from the text.",
            "point of minutes": "Generate a concise point of minutes (PoM) for the following text.",
            "custom": "Follow the user's custom instructions for the given text."
        }
        user_prompt = f"{task_map[task]}: {text}"
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": user_prompt}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error accessing Groq API: {e}"

def display_text_processing():
    st.write("### Text Processing ")
    task = st.selectbox("Select Task", ["summary", "highlight", "point of minutes", "custom"], index=0)
    user_input = st.text_area("Enter your text or article:", key='llama_input', height=200)
    if st.button("Process") and user_input:
        with st.spinner("Processing..."):
            output = process_text_with_groq(task, user_input)
            st.success(output)

# Part 2: Document Search
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
    st.write("### Document Search")
    query = st.text_input("Ask a question about your document:")
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.success(result["answer"])
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)

# Part 3: Dedicated Chat Window
def chat_with_groq(text):
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error accessing Groq API: {e}"

def display_chat_window():
    st.write("### Chat with Assistant")
    user_input = st.text_input("Enter your message:")
    if st.button("Send") and user_input:
        with st.spinner("Generating response..."):
            output = chat_with_groq(user_input)
            st.success(output)

# Main Functionality
def main():
    st.sidebar.title("App Navigation")
    app_mode = st.sidebar.radio("Choose a mode", ["Introduction", "Text Processing", "InsightDoc Analyzer", "Chat"])
    initialize_session_state()
    if app_mode == "Introduction":
        display_introduction()
    elif app_mode == "Text Processing":
        display_text_processing()
    elif app_mode == "InsightDoc Analyzer":
        uploaded_files = st.sidebar.file_uploader("Upload PDF files:", accept_multiple_files=True)
        if uploaded_files:
            text = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    text.append(temp_file.name)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
            chunks = text_splitter.split_documents(text)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            chain = create_conversational_chain(vector_store)
            display_document_search(chain)
    elif app_mode == "Chat":
        display_chat_window()

if __name__ == "__main__":
    main()
