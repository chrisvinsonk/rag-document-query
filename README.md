# Document Q&A with RAG

This project implements a Retrieval Augmented Generation (RAG) system using Streamlit, Google's Gemini AI, and FAISS for efficient similarity search. It allows users to upload PDF documents, ask questions about the content, and receive AI-generated answers.

## Features

- PDF document upload and processing
- Text chunking and vector store creation using FAISS
- AI-powered question answering using Google's Gemini model
- User-friendly Streamlit interface
- Response download as a formatted PDF

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/document-qa-rag.git
   cd document-qa-rag
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run rag_app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Upload a PDF document using the file uploader.

4. Wait for the document to be processed.

5. Enter your question about the document in the text input field.

6. View the AI-generated answer.

7. Optionally, download the response as a PDF.

## Dependencies

- streamlit
- google-generativeai
- PyPDF2
- langchain
- faiss-cpu
- sentence-transformers
- torch
- python-dotenv
- reportlab

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.