def generate_response(query, relevant_chunks, openai_client):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the "
        "following pieces of retrieved context to answer the question. If "
        "you don't know the answer, say that you don't know. Use three "
        "sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + query
    )
    response = openai_client.responses.create(model="gpt-5.4", input=prompt)

    return response


def generate_query_answer(query, openai_client):
    prompt = (
        "Generate a concise, factual answer to the following question as it "
        "might appear in a Wikipedia article about large language models. "
        "Use precise technical terminology and avoid speculation."
        "\n\nQuestion:\n" + query
    )
    response = openai_client.responses.create(model="gpt-5.4", input=prompt)

    return response
