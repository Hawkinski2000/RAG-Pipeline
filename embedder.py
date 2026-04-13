def get_embeddings_batch(texts, openai_client):
    response = openai_client.embeddings.create(
        input=texts, model="text-embedding-3-large"
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings
