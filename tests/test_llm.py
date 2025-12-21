from unittest.mock import patch

from multidoc_rag.llm import LLMClient


class FakeDoc:
    def __init__(self, text):
        self.page_content = text


class DummyLLM:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, messages):
        class Resp:
            content = "dummy answer"

        return Resp()


def test_llm_answer_with_get_relevant_documents(monkeypatch):
    # Patch ChatGroq so creating LLMClient won't fail in test env
    monkeypatch.setattr("multidoc_rag.llm.ChatGroq", DummyLLM)

    class Retriever:
        def get_relevant_documents(self, q):
            return [FakeDoc("this is context")]

    client = LLMClient(model="x", temperature=0)
    ans = client.answer(Retriever(), "What is X?")
    assert "dummy answer" in ans
