from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pytesseract
import cv2
import numpy as np
from PIL import Image
import os
import io
import uuid
import shutil
from typing import List, Optional
import uvicorn

# RAG components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

# Initialize FastAPI app
app = FastAPI(title="Interactive Text Extractor with RAG")

# Configure paths
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
VECTOR_STORE_PATH = "static/vector_stores"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global variable to store the most recent extracted text
current_extracted_text = ""
current_vector_store = None

# Initialize RAG components
def initialize_rag_components():
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Initialize embeddings model - using a lightweight model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # Initialize language model for answering questions
    # Using a smaller model for faster inference
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    llm_pipeline = pipeline(
    "text2text-generation",  # ✅ CORRECT PIPELINE TYPE for T5
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
    )

    
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    
    return text_splitter, embeddings, llm

# Initialize RAG components
text_splitter, embeddings, llm = initialize_rag_components()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    global current_extracted_text, current_vector_store
    
    # Generate a unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the image
    try:
        # Open image with PIL
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Use Tesseract for OCR
        boxes = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        # Prepare text and bounding boxes for the response
        recognized_data = []
        full_text = []
        
        for i in range(len(boxes['text'])):
            if int(boxes['conf'][i]) > 60 and boxes['text'][i].strip():  # Only confident detections with non-empty text
                text = boxes['text'][i]
                (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
                
                # Add to response data
                recognized_data.append({
                    'text': text,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })
                
                full_text.append(text)
                
                # Draw rectangle on image (for visualization)
                cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save the processed image
        output_filename = f"output_{unique_filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, img_cv)
        
        # Combine all extracted text
        current_extracted_text = " ".join(full_text)
        
        # Create vector store for RAG
        if current_extracted_text:
            # Split text into chunks
            text_chunks = text_splitter.split_text(current_extracted_text)
            
            # Create vector store
            vector_store_path = os.path.join(VECTOR_STORE_PATH, f"vs_{uuid.uuid4()}")
            current_vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            
            # Save vector store
            current_vector_store.save_local(vector_store_path)
        
        return JSONResponse({
            'data': recognized_data, 
            'output_image': f"/static/outputs/{output_filename}",
            'full_text': current_extracted_text
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )

@app.post("/query")
async def query_text(question: str = Form(...)):
    global current_vector_store
    
    if not current_vector_store or not current_extracted_text:
        return JSONResponse({
            'answer': "No text has been extracted yet. Please upload an image first."
        })
    
    try:
        # Create a retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=current_vector_store.as_retriever(search_kwargs={"k": 3})
        )
        
        # Run the query
        response = qa_chain.run(question)
        
        return JSONResponse({
            'answer': response,
            'context': current_extracted_text[:300] + "..." if len(current_extracted_text) > 300 else current_extracted_text
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during query: {str(e)}"}
        )

@app.get("/extracted_text")
async def get_extracted_text():
    global current_extracted_text
    return JSONResponse({'text': current_extracted_text})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
