"""
src/generation/prompt.py
────────────────────────
Prompt templates used by the generation chain.

Design principles:
- Ground the answer strictly in retrieved context
- Always cite sources
- Return a controlled fallback when context is insufficient
- Prevent prompt injection from document content
"""

from langchain_core.prompts import ChatPromptTemplate

# ── System message ────────────────────────────────────────────────────────

SYSTEM_MESSAGE = """You are an enterprise knowledge assistant.
Your job is to answer questions using ONLY the document excerpts provided below.

Rules you must follow:
1. Answer using ONLY information from the provided context.
2. If the context does not contain enough information, respond with:
   "I could not find a reliable answer in the indexed documents."
3. Always cite your sources at the end of your answer using the format:
   Sources: [file_name, page/section]
4. Do not fabricate facts, statistics, or references.
5. If the question is ambiguous, state your interpretation before answering.
6. Keep your answer concise and professional.

--- CONTEXT START ---
{context}
--- CONTEXT END ---
"""

# ── Human message ─────────────────────────────────────────────────────────

HUMAN_MESSAGE = "Question: {question}"

# ── Assembled prompt ──────────────────────────────────────────────────────

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("human", HUMAN_MESSAGE),
    ]
)


def format_context(documents) -> str:
    """
    Format retrieved documents into the context string injected into the prompt.

    Each chunk is separated and labelled with its source metadata so the
    model can cite it accurately.
    """
    parts = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        source_label = (
            f"[{i}] {meta.get('file_name', 'unknown')} "
            f"(page {meta.get('page', meta.get('chunk_index', '?'))})"
        )
        parts.append(f"{source_label}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts) if parts else "No relevant documents found."
