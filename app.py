import os
import tempfile
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents.base import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever
#from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.buffer_window import ConversationBufferWindowMemory

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredPDFLoader
import _thread
from langchain import HuggingFaceHub

# Function to parse PDFs
@st.cache_data(show_spinner=True)
def parse_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()
    return docs

# Function to embed text into vector store
@st.cache_data(show_spinner=True)
def embed_text(_documents):
    text_splitter = CharacterTextSplitter(
        separator="\n\nQ:",
        chunk_size=100,
        chunk_overlap=10,
        length_function=len,
        is_separator_regex=False,
    )

    split_text = []
    for doc in _documents:
        doc_content = doc.page_content
        doc_obj = Document(page_content=doc_content)
        split_text.extend(text_splitter.split_documents([doc_obj]))

    # Extract text from Document objects
    split_text_content = [doc.page_content for doc in split_text]

    embeddings = HuggingFaceEmbeddings()
    index = FAISS.from_texts(split_text_content, embeddings)
    return index

# Function to create buffer memory
@st.cache_data(show_spinner=True)
def create_buffer_memory():
    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
    return memory

# Function to create LLM using HuggingFaceHub
@st.cache_resource(show_spinner=True, hash_funcs={_thread.RLock: lambda _: None})
def create_llm():
    api_key = st.secrets["HUGGINGFACE_API_KEY"]
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base", 
        model_kwargs={"temperature": 1, "max_length": 2048, "repetition_penalty": 1.2}, huggingfacehub_api_token=api_key)
    return llm

# Function to get answer using ConversationalRetrievalChain
def get_answer(llm, retriever, memory, query):
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
    )
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Modified prompt
    prompt = (
        "Use the following pieces of context to answer the question at the end. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
        "Provide a detailed and complete answer using the full context provided."
        # f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    response = conversation_chain.run(prompt)
    return response

# Streamlit App
st.header("FAQ QnA")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    docs = parse_pdf(temp_file_path)
    faiss = embed_text(docs)
    retriever = VectorStoreRetriever(vectorstore=faiss, search_kwargs={"k": 3})
    memory = create_buffer_memory()
    llm = create_llm()

    query = st.text_area("Ask Aramex FAQ Bot")
    button = st.button("Submit")
    if button:
        answer = get_answer(llm, retriever, memory, query)
        st.write(f"Question: {query}")
        st.write(f"Answer: {answer}")

    os.remove(temp_file_path)
