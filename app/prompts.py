from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
Use the given context to answer the question.
If the answer is not present, say "I don't know".

Context:
{context}
"""

def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])
