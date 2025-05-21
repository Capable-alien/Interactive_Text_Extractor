# Interactive Text Extractor with RAG

A web application that allows users to upload images containing text and interactively select, copy, and extract the recognized text. The application uses Optical Character Recognition (OCR) technology, powered by Tesseract, to identify and overlay selectable text boxes on the image. Additionally, it implements a Retrieval-Augmented Generation (RAG) system to allow users to ask questions about the extracted text.

## Features

* **Text Recognition**: Extracts text from uploaded images using Tesseract OCR.
* **Interactive Overlay**: Displays the recognized text as selectable overlays on the image.
* **Text Selection and Copy**: Users can select and copy recognized text directly from the image.
* **RAG Integration**: Ask questions about the extracted text and get AI-generated answers.

## Technologies Used

* **FastAPI**: Modern, fast web framework for building APIs with Python
* **Tesseract OCR**: Text recognition from images
* **OpenCV**: Image processing
* **LangChain**: Framework for developing applications powered by language models
* **FAISS**: Vector storage for similarity search
* **HuggingFace Transformers**: For embeddings and language models
* **HTML, CSS, JavaScript**: Frontend interface with interactive elements

Tesseract OCR is installed from the link,
https://github.com/UB-Mannheim/tesseract/wiki

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/interactive-text-extractor-with-rag.git
   cd interactive-text-extractor-with-rag
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure Tesseract is installed and update the path in app.py if necessary:
   ```python
   # Update this line with your Tesseract installation path if different
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Use the application:
   - Upload an image containing text
   - Wait for text recognition to complete
   - Interact with the recognized text overlays
   - Ask questions about the extracted text using the RAG system

## How the RAG is implemented

1. **Text Extraction**: When an image is uploaded, text is extracted using Tesseract OCR.
2. **Text Processing**: The extracted text is split into chunks.
3. **Vector Embedding**: Text chunks are converted to vector embeddings using Sentence Transformers.
4. **Vector Storage**: Embeddings are stored in a FAISS vector store for efficient similarity search.
5. **Query Processing**: When a question is asked, the system retrieves relevant text chunks.
6. **Answer Generation**: A language model uses the retrieved chunks to generate an answer to the question.

## Customization

- **Language Models**: Replace the default language model with a more powerful one for better RAG capabilities.
- **Embeddings**: Different embedding models can be used for better semantic understanding.
