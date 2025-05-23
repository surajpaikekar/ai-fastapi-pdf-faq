# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import fitz  # PyMuPDF
import torch
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

app = FastAPI()

UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vectorstores"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Load sentence transformer model for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load LLaMA or similar model (replace with any HF instruction model)
device = "cuda" if torch.cuda.is_available() else "cpu"
llm_pipeline = pipeline(
    # "text-generation",
    # model="tiiuae/falcon-7b-instruct",  # swap with LLaMA if you have GPU
    # model="EleutherAI/gpt-neo-125M",  # ~500MB
    "text2text-generation",
    model="google/flan-t5-base",
    device=0 if device == "cuda" else -1,
    max_new_tokens=256,
    temperature=0.7,
    eos_token_id= None # or set to 50256  # or define custom stop sequence depending on model
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Helper: Extract text from PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Helper: Split text and create embeddings
def create_vector_store(text, pdf_id):
    # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(text)
    # Create FAISS vector store
    vectorstore = FAISS.from_texts(texts, embedding=embedding_model)
    vectorstore.save_local(f"{VECTOR_FOLDER}/{pdf_id}")
    return vectorstore

# Helper: Load vector store
def load_vector_store(pdf_id):
    vector_path = os.path.join(VECTOR_FOLDER, pdf_id)
    print(f"Trying to load vector store from: {vector_path}")
    print("Files in vectorstore path:", os.listdir(vector_path))
    return FAISS.load_local(
        vector_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # ✅ add this line
    )



@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    # Remove .pdf and replace spaces with underscores
    base_filename = os.path.splitext(file.filename)[0].replace(" ", "_")
    file_path = os.path.join(UPLOAD_FOLDER, base_filename + ".pdf")
    
    # with open(file_path, "wb") as f:
    #     f.write(await file.read())

    async with file as f:
        with open(file_path, "wb") as out_file:
            out_file.write(await f.read())

    text = extract_text(file_path)
    print(f"Text length: {len(text)}")

    try:
        create_vector_store(text, pdf_id=base_filename)
        print("Vector store created successfully")
    except Exception as e:
        print("Vector store creation failed:", e)
    
    return {"message": "PDF uploaded and processed", "pdf_id": base_filename}


# Ask a question using RAG from stored PDF vector DB
@app.post("/ask/")
async def ask_question(pdf_id: str = Form(...), question: str = Form(...)):
    vectorstore_path = os.path.join(VECTOR_FOLDER, pdf_id)
    print(f"Looking for vector store at: {vectorstore_path}")

    if not os.path.exists(vectorstore_path):
        return JSONResponse(status_code=404, content={"error": f"Vector store not found at {vectorstore_path}"})

    try:
        vectorstore = load_vector_store(pdf_id)
    except Exception as e:
        print("Error loading vectorstore:", e)
        return JSONResponse(status_code=500, content={"error": "Failed to load vector store"})

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    prompt_template = """
    Use the following extracted parts of a long document to answer the question.
    Only use the factual information from the context. 
    If you don’t know the answer, just say “I don’t know.” Don’t try to make up an answer.

    Context:
    {context}

    Question: {question}
    Helpful Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    # result = qa_chain.run(question)
    # return {"answer": result}
    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }


