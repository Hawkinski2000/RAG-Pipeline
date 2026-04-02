def build_qa_prompt(chunks):
    formatted_chunks = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks))

    return f"""
You are generating a high-quality question-answer pair dataset for retrieval evaluation.

Each item is based on exactly ONE text chunk.

Task:
1. First choose ONE chunk index from the list.
2. Write a natural question that can be answered using ONLY that chunk.
3. Write a correct answer based ONLY on that chunk.

Rules:
- You MUST choose a valid index from 0 to {len(chunks)-1}.
- Do NOT use outside knowledge.
- Do NOT combine multiple chunks.
- Do NOT mention or refer to "the chunk" in the question or answer.
- The question should sound natural, like a real exam or interview question.
- The answer must be fully supported by the selected chunk only.
- Keep the answer concise (1–4 sentences).

Question quality:
- Prefer "why", "how", or explanatory questions.
- Avoid simple copy-paste questions.

Output rules:
- Return ONLY raw JSON.
- Your entire response must be a single valid JSON object.
- Do NOT include markdown, code blocks, or any extra text.

Return format:

{{
  "chunk_index": int,
  "question": string,
  "answer": string
}}

Chunks:
{formatted_chunks}
"""
