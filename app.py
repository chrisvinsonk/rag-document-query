# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import GooglePalmEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.llms import GooglePalm
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# def process_pdf(file):
#     pdf_reader = PdfReader(file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# def split_text(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def create_vector_store(text_chunks):
#     embeddings = GooglePalmEmbeddings()
#     vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vector_store

# def get_response(vector_store, query):
#     llm = GooglePalm()
#     qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
#     response = qa_chain.run(query)
#     return response

# st.set_page_config(page_title="RAG Document Q&A", page_icon="📚")

# st.title("📚 RAG Document Q&A")

# uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

# if uploaded_file is not None:
#     text = process_pdf(uploaded_file)
#     text_chunks = split_text(text)
#     vector_store = create_vector_store(text_chunks)
    
#     st.success("Document processed successfully!")
    
#     query = st.text_input("Enter your question about the document:")
    
#     if query:
#         response = get_response(vector_store, query)
        
#         st.subheader("Answer:")
#         st.write(response)
        
#         # Download button for the response
#         st.download_button(
#             label="Download Response",
#             data=response,
#             file_name="response.txt",
#             mime="text/plain"
#         )
# else:
#     st.info("Please upload a PDF document to get started.")

# st.sidebar.title("About")
# st.sidebar.info(
#     "This app uses Retrieval Augmented Generation (RAG) to answer questions about uploaded PDF documents. "
#     "It leverages Google's PaLM API and FAISS for efficient vector storage and retrieval."
# )

import streamlit as st
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Set page config
st.set_page_config(page_title="Document Query with RAG", page_icon="📚")

# Load environment variables
load_dotenv()

# Set up Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Function to process the uploaded PDF
def process_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

# Function to get response from Gemini
def get_gemini_response(query, vector_store):
    docs = vector_store.similarity_search(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

# Function to create a PDF
def create_pdf(query, response):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add title
    story.append(Paragraph("Document Q&A Response", styles['Title']))
    story.append(Spacer(1, 12))

    # Add query
    story.append(Paragraph("Question:", styles['Heading2']))
    story.append(Paragraph(query, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Add response
    story.append(Paragraph("Answer:", styles['Heading2']))
    story.append(Paragraph(response, styles['BodyText']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Streamlit UI
st.title("Document Query with RAG")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Initialize session state for vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if uploaded_file is not None:
    if st.session_state.vector_store is None:
        with st.spinner('Processing your file... This may take a moment.'):
            text = process_pdf(uploaded_file)
            text_chunks = split_text(text)
            st.session_state.vector_store = create_vector_store(text_chunks)
        st.success("Document processed successfully!")
    
    query = st.text_input("Enter your question about the document:")
    
    if query:
        with st.spinner('Generating response...'):
            response = get_gemini_response(query, st.session_state.vector_store)
        st.write("Answer:")
        st.write(response)
        
        # Create and download PDF
        pdf_buffer = create_pdf(query, response)
        st.download_button(
            label="Download Response as PDF",
            data=pdf_buffer,
            file_name="response.pdf",
            mime="application/pdf"
        )
else:
    st.session_state.vector_store = None

# Instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Upload a PDF document using the file uploader.
2. Once uploading and processing is complete, enter your question about the document in the text input field.
""")

# Run the Streamlit app
if __name__ == "__main__":
    pass