

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.rag_pipeline import run_rag_pipeline_with_context

from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper


ollama_llm = ChatOllama(model="llama3",temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



evaluator_llm = LangchainLLMWrapper(ollama_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)



# fn to load evaluation data 
def load_evaluation_data():
    file_path = "/Users/syednizamuddin/Documents/GenAI_projects/retail-digital-rag-assistant/evaluation/test_questions.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def build_evaluation_dataset(test_cases):

    rows = []
    for test_case in test_cases:

        #get the question & ground truth(expected answer)
        question = test_case['question']
        ground_truth = test_case['ground_truth']

        #get the context & the llm generated answer form rag pipelein 
        answer , contexts = run_rag_pipeline_with_context(question)
        row = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth, 
        }

        rows.append(row)

    return Dataset.from_list(rows)


if __name__ == "__main__":
    test_cases = load_evaluation_data()
    dataset = build_evaluation_dataset(test_cases)
    result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ],
    llm=evaluator_llm, embeddings=evaluator_embeddings
)

    print(result)

    






