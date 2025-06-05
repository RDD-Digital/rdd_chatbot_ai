import os
from langdetect import detect  # ⬅️ Per rilevare la lingua automaticamente

os.environ["LANGCHAIN_FAISS_ENABLE_DANGEROUS_DESERIALIZATION"] = "1"

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains import LLMChain

from pathlib import Path

# FastAPI setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Embeddings e modello LLM
embedding = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3")

# Prompt con riconoscimento lingua
prompt = PromptTemplate.from_template("""
Rispondi sempre in questa lingua: {language}

Domanda: {question}
Contesto: {context}

Devi rispondere in maniera sintetica, a meno che non vengano richieste informazioni specifiche.
Analizza correttamente la domanda prima di rispondere.
Non inventare nulla: rispondi solo in base ai documenti disponibili.
Non fare mai supposizioni. Se non conosci la risposta, dillo chiaramente.
Se necassario traduci in inglese corretto grammaticalmete le informazioni dei documenti.

Sei un assistente AI di RDD Italia. Rispondi in modo informale ma professionale.
Tu non sei nessuno del personale.
Non chiamare il cliente per nome.
Non parlare di argomenti diversi da RDD: se ti viene chiesto altro, rispondi che non sei autorizzato a parlarne.
Se ti chiedono chi sei, rispondi brevemente che sei un assistente virtuale.

Se devi fare un elenco, vai a capo ogni volta.
Evita di allungare il discorso con frasi inutili.

Risposta:
""")

# Caricamento o aggiornamento del vectorstore
vectorstore_path = "vectorstore"
qa_chain = None
retriever = None

def create_or_update_index():
    documents = []
    txt_loader = DirectoryLoader("docs", glob="**/*.txt")
    documents += txt_loader.load()

    for pdf_path in Path("docs").rglob("*.pdf"):
        pdf_loader = PyPDFLoader(str(pdf_path))
        documents += pdf_loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    db = FAISS.from_documents(texts, embedding)
    db.save_local(vectorstore_path)
    print("✅ Vector store aggiornato con documenti .txt e .pdf.")

create_or_update_index()

try:
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()
    qa_chain = LLMChain(llm=llm, prompt=prompt)
    print("✅ Vector store caricato correttamente.")
except Exception as e:
    print(f"❌ Errore nel caricamento del vectorstore: {e}")

# Rotte FastAPI
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat_interface.html", {"request": request})

@app.post("/chat", response_class=JSONResponse)
async def chat(
    request: Request,
    question: str = Form(...),
    context: str = Form(""),
    lang: str = Form("it")
):
    saluti = ["ciao", "salve", "buongiorno", "hey", "ehi", "hola"]
    if question.lower().strip() in saluti:
        return {"response": "Ciao! Sono l’assistente AI di RDD Italia. Come posso aiutarti oggi?"}

    try:
        detected_lang = detect(question)
    except Exception:
        detected_lang = "it"

    try:
        docs = retriever.invoke(question)
        combined_context = "\n\n".join(doc.page_content for doc in docs)
        response = qa_chain.invoke({
            "question": question,
            "context": combined_context,
            "language": lang or detected_lang
        })
        return {"response": response}
    except Exception as e:
        return {"response": f"Errore nell'elaborazione della risposta: {e}"}
