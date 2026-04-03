def build_qa_prompt(chunks):
    formatted_chunks = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks))

    return f"""
You are generating a high-quality question-answer dataset for retrieval evaluation.

Each example is based on ONE OR MORE chunks from a single Wikipedia page.

Task:
1. Select ALL chunks that contain information needed to fully answer the question.
2. Write a natural question that can be answered using those chunks.
3. Write a correct answer supported ONLY by the selected chunks.

Rules:
- You MUST choose valid indices from 0 to {len(chunks)-1}.
- You SHOULD include all chunks that are relevant, even if partially relevant.
- Multi-chunk answers are preferred when information is distributed across sections.
- Do NOT use outside knowledge.
- Do NOT include irrelevant chunks.
- The answer must be fully supported by the selected chunks only.
- Do NOT mention or refer to "chunks" in the question or answer.
- Keep the answer concise (1–4 sentences).

Question quality:
- Prefer "how", "why", or explanatory questions.
- The question should require combining information across multiple chunks when possible.
- Avoid overly narrow factual questions unless the page is simple.

Important:
- It is better to include an extra relevant chunk than to omit one.

Chunks:
{formatted_chunks}
"""
