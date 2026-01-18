def get_retriever(vectorstore, k=4):
    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
