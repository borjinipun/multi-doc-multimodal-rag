from multidoc_rag.utils import split_documents


class FakeDoc:
    def __init__(self, text: str):
        self.page_content = text


def test_split_documents_creates_chunks():
    docs = [FakeDoc("a" * 5000)]
    chunks = split_documents(docs, chunk_size=2000, chunk_overlap=0)
    assert len(chunks) >= 2
    # each chunk should have page_content attribute
    assert hasattr(chunks[0], "page_content")
