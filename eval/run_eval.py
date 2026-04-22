from dotenv import load_dotenv

load_dotenv()

import json
import os
import argparse
import time
from qdrant_client import QdrantClient
from openai import OpenAI
from tqdm import tqdm
from retriever import query_documents
from reranker import rerank_chunks
from generator import generate_response, generate_query_answer
from .prompts.faithfulness import build_faithfulness_prompt
from .tools import compute_faithfulness_tool


DATASET_DIR = "wiki_eval_v4"
TOP_K = 20
TOP_N = 3
MODEL = "gpt-5.4-nano"
MAX_OUTPUT_TOKENS = 500


def get_next_experiment_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(script_dir, "experiments")
    os.makedirs(exp_dir, exist_ok=True)

    existing = [d for d in os.listdir(exp_dir) if d.startswith("exp_")]

    if not existing:
        version = 1
    else:
        versions = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
        version = max(versions) + 1

    exp_name = f"exp_{version}"
    exp_path = os.path.join(exp_dir, exp_name)

    os.makedirs(exp_path, exist_ok=True)

    return exp_path, exp_name


def compute_metrics(chunks, gt_set):
    chunks_set = set((chunk["title"], chunk["chunk_index"]) for chunk in chunks)

    precision = len(chunks_set & gt_set) / len(chunks_set) if chunks_set else 0
    recall = len(chunks_set & gt_set) / len(gt_set) if gt_set else 0
    hit = int(len(chunks_set & gt_set) > 0)

    mrr = 0.0
    for i, chunk in enumerate(chunks):
        if (chunk["title"], chunk["chunk_index"]) in gt_set:
            mrr = 1 / (i + 1)
            break

    return {
        "precision": precision,
        "recall": recall,
        "hit": hit,
        "mrr": mrr,
    }


def compute_faithfulness(query, answer, chunks, openai_client):
    prompt = build_faithfulness_prompt(query, answer, chunks)

    while True:
        try:
            response = openai_client.responses.create(
                model=MODEL,
                input=prompt,
                tools=[compute_faithfulness_tool],
                tool_choice={"type": "function", "name": "compute_faithfulness"},
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

            arguments = response.output[0].arguments
            args = json.loads(arguments)

            return args

        except Exception as e:
            print(f"retrying compute_faithfulness ({e})")
            time.sleep(1)


def run_eval(num_examples, description):
    exp_path, exp_name = get_next_experiment_path()
    trace_path = os.path.join(exp_path, "trace.jsonl")
    results_path = os.path.join(exp_path, "results.json")

    trace_file = open(trace_path, "a", encoding="utf-8")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, "datasets", DATASET_DIR)

    client = QdrantClient(url="http://localhost:6333")
    openai_client = OpenAI()

    results = []
    with open(f"{datasets_dir}/data.jsonl", "r", encoding="utf-8") as f:
        if num_examples is None:
            total = sum(1 for _ in f)
            f.seek(0)
        else:
            total = num_examples

        for i, line in enumerate(
            tqdm(
                f,
                total=total,
                desc=f"Evaluating on {DATASET_DIR}",
                unit="examples",
            )
        ):
            if i >= total:
                break

            example = json.loads(line)

            query = example["question"]
            gt_title = example["title"]
            gt_set = set((gt_title, idx) for idx in example["chunk_indices"])

            # query_answer = generate_query_answer(query, openai_client).output_text
            # expanded_query = f"{query}\n{query_answer}"

            retrieved_chunks = query_documents(query, client, openai_client, TOP_K)
            # retrieved_chunks = query_documents(
            #     expanded_query, client, openai_client, TOP_K
            # )
            reranked_chunks = rerank_chunks(query, retrieved_chunks, TOP_N)

            retrieved_metrics = compute_metrics(retrieved_chunks, gt_set)
            reranked_metrics = compute_metrics(reranked_chunks, gt_set)

            response = generate_response(query, reranked_chunks, openai_client)

            answer = response.output_text
            faithfulness = compute_faithfulness(
                query, answer, reranked_chunks, openai_client
            )

            results.append(
                {
                    "precision_k": retrieved_metrics["precision"],
                    "recall_k": retrieved_metrics["recall"],
                    "hit_k": retrieved_metrics["hit"],
                    "mrr_k": retrieved_metrics["mrr"],
                    "precision_n": reranked_metrics["precision"],
                    "recall_n": reranked_metrics["recall"],
                    "hit_n": reranked_metrics["hit"],
                    "mrr_n": reranked_metrics["mrr"],
                    "faithfulness": faithfulness["score"],
                }
            )

            trace_row = {
                "query": query,
                # "expanded_query": expanded_query,
                "gt": {
                    "title": example["title"],
                    "chunk_indices": example["chunk_indices"],
                },
                "retrieved_chunks": [
                    {"title": chunk["title"], "chunk_index": chunk["chunk_index"]}
                    for chunk in retrieved_chunks
                ],
                "reranked_chunks": [
                    {"title": chunk["title"], "chunk_index": chunk["chunk_index"]}
                    for chunk in reranked_chunks
                ],
                "answer": answer,
                "faithfulness": faithfulness,
            }

            trace_file.write(json.dumps(trace_row) + "\n")
            trace_file.flush()

        trace_file.close()

    summary = {
        key: sum(result[key] for result in results) / len(results)
        for key in results[0].keys()
    }

    exp_results = {
        "experiment": exp_name,
        "dataset": DATASET_DIR,
        "num_examples": num_examples or total,
        "description": description,
        "top_k": TOP_K,
        "top_n": TOP_N,
        "summary": summary,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(exp_results, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--desc", type=str, default="")
    args = parser.parse_args()

    summary = run_eval(args.num_examples, args.desc)
    print(summary)
