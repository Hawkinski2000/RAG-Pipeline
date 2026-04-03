import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from crawler import crawl
from chunker import split_text
from .prompts.qa_generation import build_qa_prompt
from .tools import generate_qa_pair_tool


MAX_LINKS = 200
MODEL = "gpt-5.4-nano"
PRICING = {"input": 0.20, "output": 1.25}
MAX_OUTPUT_TOKENS = 500

load_dotenv()


def get_next_dataset_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    existing = [d for d in os.listdir(datasets_dir) if d.startswith("wiki_eval_v")]

    if not existing:
        version = 1
    else:
        versions = [
            int(d.split("_v")[-1]) for d in existing if d.split("_v")[-1].isdigit()
        ]
        version = max(versions) + 1

    dataset_name = f"wiki_eval_v{version}"
    dataset_path = os.path.join(datasets_dir, dataset_name)

    os.makedirs(dataset_path, exist_ok=True)

    return dataset_path, dataset_name


def generate_dataset(seed, max_links):
    dataset_path, dataset_name = get_next_dataset_path()
    data_path = os.path.join(dataset_path, "data.jsonl")
    meta_path = os.path.join(dataset_path, "meta.json")

    openai_client = OpenAI()

    corpus = crawl(seed=seed, max_links=max_links)

    total_examples = 0
    total_chunks = 0
    total_cost = 0.0

    with open(data_path, "a", encoding="utf-8") as f:
        for page in tqdm(corpus, desc="Generating dataset", unit="pages"):
            chunks = split_text(page["text"])
            total_chunks += len(chunks)

            prompt = build_qa_prompt(chunks[:50])

            response = openai_client.responses.create(
                model=MODEL,
                input=prompt,
                tools=[generate_qa_pair_tool],
                tool_choice={"type": "function", "name": "generate_qa_pair"},
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            page_cost = (
                input_tokens * PRICING["input"] + output_tokens * PRICING["output"]
            ) / 1000000
            total_cost += page_cost
            tqdm.write(f"  Total cost: ${total_cost:.4f}")

            arguments = response.output[0].arguments
            args = json.loads(arguments)

            chunk_indices = args["chunk_indices"]
            question = args["question"]
            answer = args["answer"]

            for chunk_index in chunk_indices:
                if chunk_index < 0 or chunk_index >= len(chunks):
                    continue

            row = {
                "title": page["title"],
                "chunk_indices": chunk_indices,
                "question": question,
                "answer": answer,
                "chunks": [chunks[i] for i in chunk_indices],
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_examples += 1

    meta = {
        "name": dataset_name,
        "seed": seed,
        "num_pages": len(corpus),
        "num_examples": total_examples,
        "total_chunks": total_chunks,
        "avg_chunks_per_page": round(total_chunks / len(corpus), 1),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataset saved to: {dataset_path}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    generate_dataset(seed="Large_language_model", max_links=MAX_LINKS)
