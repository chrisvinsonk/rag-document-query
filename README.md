# RAG Document Q&A

This Streamlit application implements a Retrieval Augmented Generation (RAG) system for querying PDF documents. Users can upload a PDF, ask questions about its content, and receive AI-generated answers.

## Features

- PDF document upload and processing
- Text splitting and vector store creation using FAISS
- Query processing using Google's Generative AI (Gemini)
- Result display and download functionality

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your Google API key in the `.env` file
4. Run the app: `streamlit run app.py`

## Deployment

This app is deployed on Streamlit Community Cloud. Visit [insert_your_streamlit_app_url_here] to use the application.

## Note

Make sure to keep your Google API key confidential and do not share it publicly.