
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


import re
import requests 
import json

#load the document
def load_document(path):#path #text
    with open(path,'r') as document:
        text = document.read()
    return text

#creating chunks based on the document
#get the path , load teh documt , get the gheadings , cretae a pettrn split based on patter collect the chunks
def get_chunks():
    data_path = "/Users/syednizamuddin/Documents/GenAI_projects/retail-digital-rag-assistant/data/electronic_policy_handbook.txt"
    text = load_document(data_path)

    #pattern creation & splitting the document
    headings = ["ELECTRONICS RETURN AND REPLACEMENT POLICY",
        "REFUND POLICY",
        "WARRANTY POLICY",
        "DISCOUNT AND CASHBACK POLICY",
        "DELIVERY AND SHIPPING POLICY",
        "CUSTOMER SUPPORT POLICY",
        "PAYMENT REVERSAL POLICY",
        "PROMOTIONAL CASHBACK SETTLEMENT POLICY",
        "ORDER CANCELLATION REFUND POLICY",
        "FAILED EMI TRANSACTION POLICY"]
    
    pattern = "(" + "|".join(map(re.escape, headings)) + ")"
    parts = re.split(pattern, text)

    
    chunks = []
    for i in range(1, len(parts),2):#get odd as heading , even as conteent & then append both
        heading = parts[i].strip()
        content = parts[i+1].strip()

        chunk = heading + "\n" + content
        chunks.append(chunk)
    return chunks



#create vector store & embed the chunks
#load the model , create a vector store , - takes chunks, embedding model & returns the persisten
def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    #create persistent vector store
    vector_store = Chroma.from_texts(texts = chunks, embedding =embedding_model, persist_directory = 'chromadb_store')
    return vector_store

#retrieve chunks function, pass the queery , get the chunks
def retrieve_chunks(vector_store, query, k= 3):
    relevant_chunks = vector_store.similarity_search(query = query, k = k)
    return relevant_chunks

#BM25 retreiver , get chunks do wordcount freq, query most words match get the chunk
def bm25_retriver(chunks, query, k = 3):
    #cretae BM25 retriever
    bm25_retriever = BM25Retriever.from_texts(texts = chunks, k = k)
    bm25_retriever.k = k
    relevant_chunks = bm25_retriever.invoke(query)
    return relevant_chunks


#hybrid retreival function, sum of both retrivers and final deduplication
def hybrid_retrieval(vector_store, chunks, query, k=3):
    semantic_results = retrieve_chunks(vector_store, query, k)
    bm25_results = bm25_retriver(chunks, query, k)
    combined_results = semantic_results + bm25_results

    unique_hybrid_chunks = []
    seen = set()
    for doc in combined_results:
        chunk_content = doc.page_content
        if chunk_content not in seen:
            seen.add(chunk_content)
            unique_hybrid_chunks.append(doc)

    return unique_hybrid_chunks


#reranker mechanism, 
def rerank_chunks(retrieved_chunks, query, k = 2):
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') #get the model
    pairs = [(query, chunk.page_content) for chunk in retrieved_chunks]     #make the pairs
    scores  = reranker.predict(pairs)#uisng the model get the scores 
    #sort per the scores
    scored_docs = list(zip(retrieved_chunks, scores))
    scored_docs = sorted(scored_docs,key=lambda x: x[1],reverse=True)
    reranked_docs = [doc for doc, score in scored_docs[:k]]
    return reranked_docs

#build context
def build_context(reranked_docs):
    context = ""
    for index, doc in enumerate(reranked_docs, start=1):
        context += f"\nContext {index}:\n"
        context += doc.page_content
        context += "\n"
    return context

#build prompt whch will be injeted with context , query and instructions
def build_prompt(query, context):

    prompt = f"""
    You are a helpful retail policy assistant.

    Answer the user's question using ONLY the context below.
    If the answer is not available in the context, say:
    "I don't have enough information in the provided policy documents."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    return prompt


#genrate answer using prompt
def generate_answer(prompt):
    url = "http://localhost:11434/api/generate"

    payload = {
        'model':'llama3' , 
        'prompt': prompt, 
        "stream": False 
    }

    response = requests.post(url, json=payload) 
    response_json  = response.json()
    return response_json["response"]


#make the final function which we can run or call in the backend logic
#input only query

def run_rag_pipeline(query):
    chunks = get_chunks() #document loading fn #structured chunking fn
    vector_store = create_vector_store(chunks) #vector store fn
    combined_results = hybrid_retrieval(vector_store, chunks, query, k = 3) #retrival fun, semnatic, bm25, hybrid, and final selection fn
    reranked_results = rerank_chunks(combined_results, query, k = 2)#rerankign fn
    context = build_context(reranked_results) #context building fn
    prompt = build_prompt(query, context) #prompt builder fn
    answer = generate_answer(prompt)#answer genration fn
    return answer




if __name__ == "__main__":

    # chunks = get_chunks()
    # # for chunk in chunks:
    # #     print(chunk)
    # #     print("--------------------------------------------------")
    # # print(f"Total chunks created: {len(chunks)}")
    # #create vector store 
    # vector_store = create_vector_store(chunks)
    # print("Chromdb Vector store created and persisted successfully.")

    # #query
    # query = "What is the refund timeline for prepaid orders after return approval?"
    # # results_semantic = retrieve_chunks(vector_store, query, k =3)
    # # results_bm25 = bm25_retriver(chunks, query, k = 1)
    # combined_results = hybrid_retrieval(vector_store, chunks, query, k = 3)
    # #reranked results
    # reranked_results = rerank_chunks(combined_results, query, k = 2)
    # context = build_context(reranked_results)


    # prompt = build_prompt(query, context)
    # answer = generate_answer(prompt)
    # print("Answer:" , answer)



    # # #check teh retrived results
    # # print("Combined Hybrid Retrieval Results:")
    # # for result in combined_results:
    # #     print(result.page_content)

    # # #reranker
    # # for result in reranked_results:
    # #     print(result.page_content)
    # #     print("--------------------------------------------------")

    # # print(context)



    #final funciton in the orchestartor
    query = "What is the refund timeline for prepaid orders after return approval?"
    answer = run_rag_pipeline(query)
    print("Answer:" , answer)








