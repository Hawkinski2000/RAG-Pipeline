def build_qa_prompt(chunks):
    formatted_chunks = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks))

    return f"""
You are generating a high-quality question-answer pair dataset for retrieval evaluation.

Each item is based on ONE OR MORE chunks from a single page.

Task:
1. Select one or more chunk indices from the list.
2. Write a natural question that can be answered using those chunks.
3. Write a correct answer supported ONLY by the selected chunks.

Rules:
- You MUST choose valid indices from 0 to {len(chunks)-1}.
- You MAY use multiple chunks if needed.
- Do NOT use outside knowledge.
- Do NOT include irrelevant chunks.
- The answer must be fully supported by the selected chunks only.
- Do NOT mention or refer to "chunks" in the question or answer.
- Keep the answer concise (1–4 sentences).

Question quality:
- Prefer "why", "how", or conceptual questions.
- The question should feel like something a user might naturally ask.
- Avoid copying phrases directly from the chunks.
- The question can require combining information across chunks.

Important:
- Only include chunks that are truly necessary to answer the question.

Call the function with your result.

Chunks:
{formatted_chunks}
"""
