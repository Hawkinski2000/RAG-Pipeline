from embedder import get_embeddings_batch


def query_documents(query, client, openai_client, top_k=20):
    query_embedding = get_embeddings_batch([query], openai_client)[0]

    search_result = client.query_points(
        collection_name="rag-pipeline",
        query=query_embedding,
        limit=top_k,
    ).points

    relevant_chunks = [point.payload["text"] for point in search_result]

    return relevant_chunks
