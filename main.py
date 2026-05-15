

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_pipeline import run_rag_pipeline

app = FastAPI()


class QueryRequest(BaseModel): #validation for the incoming request , field name & data type
    query:str

@app.get("/health") # routing
def health_check():
    return {"status":"ok"}

@app.post("/ask") #routing 
def ask_question(request:QueryRequest):
    query = request.query
    answer  = run_rag_pipeline(query)
    return {"query": query,"answer": answer}



