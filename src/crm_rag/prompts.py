from crm_rag.store import SearchResult

NO_ANSWER = "I don't know. The CRM records I can search do not cover this question."

SYSTEM_PROMPT = f"""You answer questions from support and account teams using CRM records.

Rules:
- Use only the facts in the CONTEXT block. Do not use outside knowledge.
- After every sentence that states a fact, cite the record it came from in square
  brackets using its id exactly as given, for example [CAS-1001] or [KB-0003].
- If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
- If records disagree, say so and cite both.
- Be concise: a short paragraph or a few bullet points."""


def format_context(results: list[SearchResult]) -> str:
    blocks = []
    for r in results:
        c = r.chunk
        blocks.append(f'<record id="{c.doc_id}" type="{c.doc_type}">\n{c.text}\n</record>')
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: list[SearchResult]) -> str:
    return f"CONTEXT:\n{format_context(results)}\n\nQUESTION: {question}"
