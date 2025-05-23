from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import re
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

app = FastAPI()

UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vectorstores"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Load sentence transformer model for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    google_api_key=api_key,
    temperature=0.7,
    convert_system_message_to_human=True
)

# Helper: Extract text page-wise
def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    page_numbers = []
    for i, page in enumerate(doc):
        page_text = page.get_text()
        if page_text.strip():
            texts.append(page_text)
            page_numbers.append(i + 1)  # 1-indexed
    return texts, page_numbers

# Helper: Create vector store with page metadata
def create_vector_store(pages, page_numbers, user_id, pdf_id):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    all_chunks = []
    all_metadata = []

    for page_text, page_num in zip(pages, page_numbers):
        chunks = splitter.split_text(page_text)
        all_chunks.extend(chunks)
        all_metadata.extend([{"page": page_num, "source": pdf_id} for _ in chunks])

    vectorstore = FAISS.from_texts(all_chunks, embedding=embedding_model, metadatas=all_metadata)

    save_path = os.path.join(VECTOR_FOLDER, user_id, pdf_id)
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    return vectorstore

# Upload PDF endpoint
@app.post("/upload/")
async def upload_pdf(user_id: str = Form(...), file: UploadFile = File(...)):
    clean_filename = file.filename.replace(" ", "_")
    pdf_id = os.path.splitext(clean_filename)[0]
    file_path = os.path.join(UPLOAD_FOLDER, clean_filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    pages, page_numbers = extract_text_by_page(file_path)
    try:
        create_vector_store(pages, page_numbers, user_id=user_id, pdf_id=pdf_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Vector store creation failed: {e}"})

    return {"message": "PDF uploaded and processed", "pdf_id": pdf_id, "user_id": user_id}

# Ask question endpoint
@app.post("/ask/")
async def ask_question(user_id: str = Form(...), pdf_id: str = Form(...), question: str = Form(...)):
    vectorstore_path = os.path.join(VECTOR_FOLDER, user_id, pdf_id)
    if not os.path.exists(vectorstore_path):
        return JSONResponse(status_code=404, content={"error": "Vector store not found"})

    try:
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to load vector store: {e}"})

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

    result = await qa_chain.ainvoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }

@app.post("/ask_web/")
async def ask_web(question: str = Form(...)):
    try:
        # Ask Gemini directly
        response = await llm.ainvoke(question)
        return {"answer": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate answer: {e}"})




@app.post("/generate-faqs/")
async def generate_faqs(user_id: str = Form(...), pdf_id: str = Form(...)):
    vectorstore_path = os.path.join(VECTOR_FOLDER, user_id, pdf_id)
    if not os.path.exists(vectorstore_path):
        return JSONResponse(status_code=404, content={"error": "Vector store not found"})

    try:
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to load vector store: {e}"})

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents("Generate FAQs from this document")
    context = "\n\n".join([doc.page_content for doc in docs])[:3000]

    prompt = f"""
Generate 10 frequently asked questions (FAQs) and their answers based on the following text content.

Text:
{context}

Format the response like:
Q1: ...
A1: ...
Q2: ...
A2: ...
...
"""

    try:
        result = await llm.ainvoke(prompt)
        raw_output = result.content

        # Parse the output into structured FAQ list
        faqs = []
        qas = re.findall(r"Q\d+:(.*?)A\d+:(.*?)(?=Q\d+:|$)", raw_output, re.DOTALL)
        for q, a in qas:
            faqs.append({
                "question": q.strip(),
                "answer": a.strip()
            })

        return JSONResponse(content={"faqs": faqs})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate FAQs: {e}"})
