from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
from crawler import crawl
from chunker import split_text
from embedder import get_embeddings_batch
from retriever import query_documents
from generator import generate_response


VECTOR_SIZE = 1536

client = QdrantClient(url="http://localhost:6333")

if client.collection_exists("rag-pipeline"):
    client.delete_collection("rag-pipeline")

client.create_collection(
    collection_name="rag-pipeline",
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)


def main():
    corpus = crawl(seed="Large_language_model", max_links=20)

    point_id = 0
    pbar = tqdm(corpus, desc="Generating embeddings", unit="page")
    for page in pbar:
        chunks = split_text(page["text"])
        embeddings = get_embeddings_batch(chunks)

        points = [
            PointStruct(
                id=point_id + i,
                vector=embeddings[i],
                payload={"text": chunks[i], "title": page["title"]},
            )
            for i in range(len(chunks))
        ]

        client.upsert(collection_name="rag-pipeline", wait=True, points=points)
        point_id += len(chunks)

    while True:
        query = input("Enter a question: ")
        relevant_chunks = query_documents(query, client)

        response = generate_response(query, relevant_chunks)

        print(f"{response.output_text}\n")


if __name__ == "__main__":
    main()
