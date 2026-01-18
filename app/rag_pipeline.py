from app.prompts import get_prompt
from app.llm import get_llm

class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = get_llm()
        self.prompt = get_prompt()

    def answer(self, query: str):
        docs = self.retriever.invoke(query)

        messages = self.prompt.format_messages(
            context=docs,
            input=query
        )

        response = self.llm.invoke(messages)

        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page_number")
            }
            for d in docs
        ]

        return {
            "answer": response.content,
            "sources": sources
        }
