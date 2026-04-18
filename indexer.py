from dotenv import load_dotenv

load_dotenv()

import os
import json
from qdrant_client import QdrantClient
from openai import OpenAI
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
from chunker import split_text
from embedder import get_embeddings_batch


CORPUS_DIR = "wiki_corpus_v1"
VECTOR_SIZE = 3072
COLLECTION_NAME = "rag-pipeline"


def load_corpus(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_index():
    print("Building index...")

    client = QdrantClient(url="http://localhost:6333")

    if client.collection_exists(COLLECTION_NAME):
        response = (
            input(f"Collection '{COLLECTION_NAME}' already exists. Overwrite? [y/N]: ")
            .strip()
            .lower()
        )

        if response != "y":
            print("Aborting index build...")
            return

        print("Deleting existing collection...")
        client.delete_collection(COLLECTION_NAME)
        print("Collection deleted.")

    print("Creating collection...")
    client.create_collection(
        collection_name="rag-pipeline",
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("Collection created.")

    openai_client = OpenAI()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.join(script_dir, "corpora", CORPUS_DIR)
    data_path = os.path.join(corpus_dir, "data.jsonl")
    meta_path = os.path.join(corpus_dir, "meta.json")

    with open(meta_path, "r") as f:
        num_pages = json.load(f)["num_pages"]

    point_id = 0
    for page in tqdm(
        load_corpus(data_path),
        total=num_pages,
        desc="Generating embeddings",
        unit="pages",
    ):
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

    print("Index built.")


if __name__ == "__main__":
    build_index()
