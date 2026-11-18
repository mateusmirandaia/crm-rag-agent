from crm_rag.chunking import chunk_document, chunk_documents, split_sentences
from crm_rag.loaders import Document, JsonDirectoryLoader


def _doc(text: str) -> Document:
    return Document(
        id="KB-9999",
        type="kb_article",
        title="Test article",
        text=text,
        header="Knowledge article, Test.",
    )


def test_short_document_is_one_chunk() -> None:
    chunks = chunk_document(_doc("One paragraph.\n\nAnother paragraph."), chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0].id == "KB-9999#0"
    assert chunks[0].body == "One paragraph. Another paragraph."


def test_every_chunk_carries_title_and_header() -> None:
    text = "\n\n".join(f"Paragraph {i} talks about topic number {i}." for i in range(30))
    chunks = chunk_document(_doc(text), chunk_size=200, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("Test article. Knowledge article, Test.\n")


def test_chunks_respect_size_budget() -> None:
    text = " ".join(f"Sentence number {i} has a few words." for i in range(200))
    for chunk in chunk_document(_doc(text), chunk_size=300, overlap=60):
        assert len(chunk.text) <= 300


def test_overlap_repeats_trailing_sentence() -> None:
    text = " ".join(f"Fact {i} is here." for i in range(40))
    chunks = chunk_document(_doc(text), chunk_size=150, overlap=40)
    first, second = split_sentences(chunks[0].body), split_sentences(chunks[1].body)
    # The next chunk opens with whole sentences from the end of the previous one.
    assert second[0] in first
    assert first[-1] in second


def test_overlong_sentence_is_hard_split() -> None:
    chunks = chunk_document(_doc("x" * 1000), chunk_size=200, overlap=0)
    assert "".join(c.body for c in chunks) == "x" * 1000


def test_sample_data_chunks_keep_metadata(settings) -> None:
    docs = list(JsonDirectoryLoader(settings.data_dir).load())
    chunks = chunk_documents(docs)
    cases = [c for c in chunks if c.doc_type == "case"]
    assert cases and all("account" in c.metadata for c in cases)
    # Denormalized account name is what lets "issues at Velmora" find its cases.
    assert any("Velmora Freight" in c.text for c in cases if c.doc_id == "CAS-1001")
