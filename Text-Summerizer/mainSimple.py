# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import fitz  # PyMuPDF
from transformers import pipeline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load QA and summarization pipeline
faq_generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Helper to extract text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# Endpoint to generate FAQs
@app.post("/generate-faqs")
# async def generate_faqs(file: UploadFile = File(...)):
async def generate_faqs(file: UploadFile = File(..., media_type='application/pdf')):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(file_path)

    prompt = f"Generate 10 FAQs based on the following text:\n{text[:3000]}"  # Limit input size
    result = faq_generator(prompt)
    return JSONResponse(content={"faqs": result[0]["generated_text"]})

# Endpoint to ask a question based on PDF
@app.post("/ask-question/")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(file_path)
    answer = qa_model(question=question, context=text)
    return JSONResponse(content={"answer": answer["answer"]})
