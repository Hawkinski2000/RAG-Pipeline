from dotenv import load_dotenv

load_dotenv()

import argparse
from qdrant_client import QdrantClient
from openai import OpenAI
from indexer import build_index
from retriever import query_documents
from bm25_retriever import BM25Retriever
from fusion import reciprocal_rank_fusion
from reranker import rerank_chunks
from generator import generate_response, generate_query_answer

MAX_LINKS = 200
TOP_K = 20
TOP_N = 3


def main(reindex=False):
    client = QdrantClient(url="http://localhost:6333")
    bm25 = BM25Retriever()

    if reindex:
        build_index(bm25)
    elif not client.collection_exists("rag-pipeline"):
        response = input("Index does not exist. Build it now? [y/N]: ").strip().lower()

        if response == "y":
            build_index(bm25)
        else:
            print("Aborting...")
            return
    else:
        bm25.load()

    openai_client = OpenAI()

    while True:
        query = input("Enter a question: ")

        query_answer = generate_query_answer(query, openai_client).output_text
        expanded_query = f"{query}\n{query_answer}"

        semantic_chunks = query_documents(expanded_query, client, openai_client, TOP_K)
        keyword_chunks = bm25.query(expanded_query, TOP_K)

        retrieved_chunks = reciprocal_rank_fusion(semantic_chunks, keyword_chunks)
        reranked_chunks = rerank_chunks(query, retrieved_chunks, TOP_N)

        response = generate_response(query, reranked_chunks, openai_client)
        print(f"Answer: {response.output_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    main(reindex=args.reindex)
