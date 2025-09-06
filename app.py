import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

# Load Groq API Key from secrets.toml
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# App Header
st.header("📄 PDF Chatbot (Groq-powered)")

with st.sidebar:
    st.title("Upload your Document")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

    # 🔽 Model selection dropdown
    model_choice = st.selectbox(
        "Choose a Groq model:",
        [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768"
        ],
        index=0  # default to "llama-3.1-8b-instant"
    )

# Extract text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings (HuggingFace, free & local)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        # Similarity search
        match = vector_store.similarity_search(user_question)

        # Define Groq LLM with selected model
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            temperature=0,
            model_name=model_choice
        )

        # QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.invoke({"input_documents": match, "question": user_question})

        st.subheader("Answer:")

        st.write(response["output_text"])
