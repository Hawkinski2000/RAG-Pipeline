MODEL = "gpt-5.4-nano"
MAX_OUTPUT_TOKENS = 500


def generate_response(query, relevant_chunks, openai_client):
    documents = [chunk["text"] for chunk in relevant_chunks]

    context = "\n\n".join(documents)
    prompt = (
        "You are an assistant for question-answering tasks. Use the "
        "following pieces of retrieved context to answer the question. If "
        "you don't know the answer, say that you don't know. Use three "
        "sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + query
    )
    response = openai_client.responses.create(
        model=MODEL, input=prompt, max_output_tokens=MAX_OUTPUT_TOKENS
    )

    return response


def generate_query_answer(query, openai_client):
    prompt = (
        "Generate a concise, factual answer to the following question as it "
        "might appear in a Wikipedia article about large language models. "
        "Use precise technical terminology and avoid speculation."
        "\n\nQuestion:\n" + query
    )
    response = openai_client.responses.create(
        model=MODEL, input=prompt, max_output_tokens=MAX_OUTPUT_TOKENS
    )

    return response
