from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
from crawler import crawl
from chunker import split_text
from embedder import get_embeddings_batch


VECTOR_SIZE = 3072


def build_index(client: QdrantClient, openai_client, seed, max_links=None):
    if client.collection_exists("rag-pipeline"):
        return

    client.create_collection(
        collection_name="rag-pipeline",
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    corpus = crawl(seed=seed, max_links=max_links)

    point_id = 0
    for page in tqdm(corpus, desc="Generating embeddings", unit="pages"):
        chunks = split_text(page["text"])
        embeddings = get_embeddings_batch(chunks, openai_client)
        points = [
            PointStruct(
                id=point_id + i,
                vector=embeddings[i],
                payload={"text": chunks[i], "title": page["title"], "chunk_index": i},
            )
            for i in range(len(chunks))
        ]
        client.upsert(collection_name="rag-pipeline", wait=True, points=points)
        point_id += len(chunks)
