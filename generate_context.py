import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from chunker import split_text
from tools import generate_chunk_context_tool
from prompts.context import build_context_prompt


CORPUS_DIR = "wiki_corpus_v1"
MODEL = "gpt-5.4-nano"
PRICING = {"input": 0.20, "output": 1.25}

load_dotenv()


def load_corpus(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def generate_context():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    context_dir = os.path.join(script_dir, "context")
    os.makedirs(context_dir, exist_ok=True)
    data_path = os.path.join(context_dir, "data.jsonl")
    meta_path = os.path.join(context_dir, "meta.json")

    corpus_dir = os.path.join(script_dir, "corpora", CORPUS_DIR)
    corpus_data_path = os.path.join(corpus_dir, "data.jsonl")
    corpus_meta_path = os.path.join(corpus_dir, "meta.json")

    with open(corpus_meta_path, "r") as f:
        corpus_meta = json.load(f)
        num_pages = corpus_meta["num_pages"]
        seed = corpus_meta["seed"]

    openai_client = OpenAI()

    total_pages = 0
    total_chunks = 0
    total_cost = 0.0

    with open(data_path, "w", encoding="utf-8") as f:
        for page in tqdm(
            load_corpus(corpus_data_path),
            total=num_pages,
            desc="Generating context",
            unit="pages",
        ):
            chunks = split_text(page["text"])

            prompt = build_context_prompt(chunks)

            while True:
                try:
                    response = openai_client.responses.create(
                        model=MODEL,
                        input=prompt,
                        tools=[generate_chunk_context_tool],
                        tool_choice={
                            "type": "function",
                            "name": "generate_chunk_context",
                        },
                    )

                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    page_cost = (
                        input_tokens * PRICING["input"]
                        + output_tokens * PRICING["output"]
                    ) / 1000000
                    total_cost += page_cost
                    tqdm.write(f"  Total cost: ${total_cost:.4f}")

                    args = json.loads(response.output[0].arguments)
                    results = args["results"]
                    assert len(results) == len(
                        chunks
                    ), f"Expected {len(chunks)} results, got {len(results)}"
                    assert all(
                        r["chunk_index"] < len(chunks) for r in results
                    ), "Out-of-bounds chunk_index in results"

                    break

                except Exception as e:
                    print(f"retrying generate_context ({e})")
                    time.sleep(1)

            for result in results:
                chunk_index = result["chunk_index"]
                chunk = chunks[chunk_index]

                row = {
                    "title": page["title"],
                    "chunk": chunk,
                    "chunk_index": chunk_index,
                    "context": result["context"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

                total_chunks += 1

            total_pages += 1

    meta = {
        "seed": seed,
        "num_pages": total_pages,
        "total_chunks": total_chunks,
        "avg_chunks_per_page": round(total_chunks / total_pages, 1),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Context saved to: {data_path}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    generate_context()
