def build_context_prompt(chunks):
    formatted = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks))

    return f"""
You are improving retrieval quality in a RAG system.

For each chunk, generate a short context sentence that explains:
- what section/topic it belongs to
- how it fits into the overall document

Rules:
- You MUST call the provided tool
- You MUST include EVERY chunk index exactly once
- Contexts must be 1-2 sentences max
- Do NOT repeat chunk text
- Do NOT omit any chunk

Chunks:
{formatted}
"""
