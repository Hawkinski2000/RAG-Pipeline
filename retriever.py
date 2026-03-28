from embedder import get_embeddings_batch


def query_documents(query, client, limit=3):
    query_embedding = get_embeddings_batch([query])[0]

    search_result = client.query_points(
        collection_name="rag-pipeline",
        query=query_embedding,
        limit=limit,
    ).points

    relevant_chunks = [point.payload["text"] for point in search_result]

    return relevant_chunks
