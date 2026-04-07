def build_faithfulness_prompt(query, answer, chunks):
    formatted_chunks = "\n\n".join(
        f"[{i}] {chunk['text']}" for i, chunk in enumerate(chunks)
    )

    return f"""
You are evaluating whether an answer is fully supported by provided context.

Question:
{query}

Answer:
{answer}

Context:
{formatted_chunks}

Task:
1. Break the answer into individual claims.
2. For each claim, determine if it is supported by the context.
3. If ANY claim is not supported, the answer is NOT faithful.
"""
