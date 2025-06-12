import os
import json
import requests
from langdetect import detect

os.environ["LANGCHAIN_FAISS_ENABLE_DANGEROUS_DESERIALIZATION"] = "1"

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# nuovo import consigliato
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains import LLMChain
from langchain.llms.base import LLM

from pathlib import Path
from typing import Optional, List

# --------- Custom GroqLLM ---------
class GroqLLM(LLM):
    model: str = "llama3-8b-8192"
    temperature: float = 0.2
    max_tokens: int = 1024
    groq_api_key: str = os.getenv("GROQ_API_KEY", "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")

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Sei l'assistente AI di RDD Italia. Rispondi solo in base al contesto fornito."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']

# --------- FastAPI setup ---------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --------- Embeddings e modello LLM ---------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = GroqLLM()

prompt = PromptTemplate.from_template("""
Rispondi sempre in questa lingua: {language}

Domanda: {question}
Contesto: {context}

system_prompt = 
Regole per l’assistente virtuale di RDD Italia:

Rispondi solo a domande su RDD Italia. Se la domanda non riguarda RDD Italia, rispondi esattamente:"Non sono autorizzato a parlare di altro."
- NON ripetere le domande dell'utente. Rispondi solamente alla domanda.
- Rispondi in modo **sintetico, diretto e professionale**.
- Verifica sempre le informazioni nei documenti forniti.
- **Non ripetere** mai la domanda dell’utente.
- **Analizza la richiesta** prima di rispondere.
- **Non aggiungere dettagli** se non richiesti.
- **Non fornire contatti** a meno che non siano esplicitamente richiesti.
- **Non inventare nulla**. Se la risposta non è nei documenti, dì: “Non ho abbastanza informazioni per rispondere.”
- Se necessario, **traduci in inglese corretto** usando un tono professionale.
- Rispondi sempre in modo **informale ma professionale**.
- **Non usare mai il nome dell’utente**.
- Se ti chiedono chi sei, rispondi: “Sono l’assistente virtuale di RDD Italia.”
- Se ti chiedono **chi è una persona**, verifica se è presente tra il personale o collaboratori. Se non è nei documenti, rispondi: “Non conosco questa persona.”
- Se ti chiedono un elenco, **crea un elenco puntato** con **un punto per riga**.
- **Evita frasi inutili**. Vai dritto al punto.

Risposta:
""")

# --------- Caricamento documenti ---------
vectorstore_path = "vectorstore"
qa_chain = None
retriever = None

def create_or_update_index():
    documents = []
    txt_loader = DirectoryLoader("docs", glob="**/*.txt")
    documents += txt_loader.load()

    from langchain.schema import Document
    for jsonl_path in Path("docs").rglob("*.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    content = data.get("completion", "").strip()
                    if content:
                        documents.append(Document(page_content=content))
                except json.JSONDecodeError:
                    continue

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

# --------- Rotte FastAPI ---------
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
        return {"response": "Dammi un minuto, sono stanco 😮‍💨"}
