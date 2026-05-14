

# #create chunks of the documets
# def get_chunks(chunk_size = 1000  , chunk_overlap = 50):#raw documnts, chunk size, chunk overlap, 
#     #get the path
#     data_path = "/Users/syednizamuddin/Documents/GenAI_projects/retail-digital-rag-assistant/data/electronic_policy_handbook.txt"
#     text = load_document(data_path)  
#     #initialize the text splitter obj
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", ".", " "])
#     #get the chunks
#     chunks  =  text_splitter.split_text(text)
#     return chunks



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
    # query = "What is the refund timeline for prepaid orders after return approval?"
    # answer = run_rag_pipeline(query)
    # print("Answer:" , answer)