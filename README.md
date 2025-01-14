# DocuBrain AI Assistant

**DocuBrain AI Assistant** is an interactive Streamlit-based application that offers powerful AI-driven text processing, document analysis, and conversational capabilities.

## Access the Application
Visit the application online at: [https://docubrain-ai-8fqgp4pgq3mwrysacuh4m4.streamlit.app/](https://docubrain-ai-8fqgp4pgq3mwrysacuh4m4.streamlit.app/)

## Demo Video

Here is the demo video: [link](https://youtu.be/ZBDC7C6HfnE).

## Features

### 1. **Text Processing**
- **Summarization**: Generate concise summaries of articles or text.
- **Highlights**: Extract key highlights from input text.
- **Points of Minutes (PoM)**: Create structured PoM from provided text.
- **Custom Instructions**: Perform tasks based on user-defined instructions.

### 2. **InsightDoc AI Analyzer( Content Engine for Document Comparison and Insights)**
#### This repository implements a Content Engine utilizing Retrieval Augmented Generation (RAG) techniques to analyze and compare multiple PDF documents, specifically Form 10-K filings from multinational companies.

#### Features
- **PDF Parsing**: Extracts and processes text from PDF documents.
- **Vectorization**: Converts text content into vectors using local embedding models.
- **Vector Store Integration**: Embeddings are stored in a vector store (Chroma or FAISS) for efficient retrieval and comparison.
- **Local LLM Integration**: A local Large Language Model (LLM), Mistral-7B-Instruct, is used for generating insights and answering queries.
- **Interactive Chatbot Interface**: Built with Streamlit, providing users a platform to query the system and obtain document insights.

#### Workflow
1. **Parse Documents**: Extract text from PDF files (e.g., Alphabet, Tesla, and Uber Form 10-K filings).
2. **Generate Embeddings**: Use Sentence-Transformers to generate dense vector embeddings for document text.
3. **Store in Vector Store**: Save embeddings in a vector store like Chroma or FAISS for fast querying.
4. **Query Engine**: Retrieve relevant information and generate insights using a local LLM.
5. **Chatbot Interface**: Users interact via a Streamlit-based chatbot UI to query and explore insights from documents.

#### Sample Queries
- "How does Tesla's automotive segment differ from its energy generation and storage segment?"
- "What are the differences in the business of Tesla and Uber?"
- "What is the total revenue for Google Search?"

### 3. **Chat with Assistant**
- Ask questions or chat with a powerful AI assistant.
- Leverage state-of-the-art models for personalized responses.

---

## Installation

### Prerequisites
Ensure the following are installed on your system:
- Python 3.8 or higher
- pip
- [Streamlit](https://docs.streamlit.io/)

### Setup Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/docubrain-ai.git
   cd docubrain-ai
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your API keys:
   ```env
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

### Part 0: Introduction
The home screen provides an overview of the app’s functionalities and links to sample files for testing.

### Part 1: Text Processing
1. Navigate to the **Text Processing** tab.
2. Choose a task: Summarization, Highlights, PoM, or Custom Instructions.
3. Input your text or article and click "Process" to get results.

### Part 2: InsightDoc AI Analyzer
1. Navigate to the **InsightDoc AI Analyzer** tab.
2. Upload one or more PDF files for analysis.
3. Perform tasks such as document comparison, summarization, or specific searches.

### Part 3: Chat Window
1. Navigate to the **Chat Window** tab.
2. Input your message in the chat box and receive intelligent responses in real-time.

---

## File Structure
```
.
├── app.py                 # Main application file
├── requirements.txt       # Required dependencies
├── .env                   # Environment variables
├── README.md              # Project documentation
├── sample_files/          # Sample files for testing
└── ...                    # Additional resources and files
```

---

## Dependencies
- **Streamlit**: For creating the web interface.
- **LangChain**: For conversational chains and document retrieval.
- **FAISS**: For vector storage and similarity search.
- **Hugging Face Transformers**: For LLM and embeddings.
- **Groq**: For specific AI-powered text processing tasks.
- **PyPDFLoader**: For PDF file loading and parsing.

---

## API Keys
- **Hugging Face**: Used for LLM-based tasks.
- **Groq**: Used for text processing tasks.

Ensure you have valid API keys for both services. Add these keys to the `.env` file as shown in the setup instructions.

---

## Sample Files
Sample PDF files are included in the `sample_files/` directory for testing purposes. Use these to explore the app’s functionalities.

---

## Contributing
Contributions are welcome! If you find a bug or want to add a feature:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact
For any questions or feedback, please reach out to [your-email@example.com](mrrahulkraggl@gmail.com).

