from qdrant_client import QdrantClient
from indexer import build_index
from retriever import query_documents
from reranker import rerank_chunks
from generator import generate_response, generate_query_answer


MAX_LINKS = 200
TOP_K = 20
TOP_N = 3


def main():
    client = QdrantClient(url="http://localhost:6333")
    build_index(client, seed="Large_language_model", max_links=MAX_LINKS)

    while True:
        query = input("Enter a question: ")

        query_answer = generate_query_answer(query).output_text
        expanded_query = f"{query}\n{query_answer}"

        relevant_chunks = query_documents(expanded_query, client, TOP_K)
        reranked_chunks = rerank_chunks(query, relevant_chunks, TOP_N)

        response = generate_response(query, reranked_chunks)
        print(f"Answer: {response.output_text}\n")


if __name__ == "__main__":
    main()
