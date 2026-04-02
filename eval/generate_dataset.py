import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from crawler import crawl
from chunker import split_text
from .prompts.qa_generation import build_qa_prompt


MAX_LINKS = 200
MODEL = "gpt-5.4-nano"
PRICING = {"input": 0.20, "output": 1.25}
MAX_OUTPUT_TOKENS = 500

load_dotenv()
openai_client = OpenAI()


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

    corpus = crawl(seed=seed, max_links=max_links)

    total_examples = 0
    total_chunks = 0
    total_cost = 0.0

    with open(data_path, "a", encoding="utf-8") as f:
        for page in tqdm(corpus, desc="Generating dataset", unit="pages"):
            chunks = split_text(page["text"])
            total_chunks += len(chunks)

            prompt = build_qa_prompt(chunks[:50])

            for attempt in range(5):
                response = openai_client.responses.create(
                    model=MODEL,
                    input=prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )

                raw = response.output_text

                try:
                    data = json.loads(raw)
                    break
                except json.JSONDecodeError as e:
                    print(f"[JSON retry {attempt+1}/5] Bad JSON:", e)
                    print("Raw output:", raw)
            else:
                print("Failed to parse JSON after 5 attempts, skipping page.")
                continue

            if not isinstance(data, dict):
                print("Model returned non-dict, skipping page.")
                continue

            required_keys = {"chunk_index", "question", "answer"}
            if not required_keys.issubset(data):
                print("Missing key(s) in response, skipping page.")
                continue

            chunk_index = data["chunk_index"]
            question = data["question"]
            answer = data["answer"]

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            page_cost = (
                input_tokens * PRICING["input"] + output_tokens * PRICING["output"]
            ) / 1000000
            total_cost += page_cost
            tqdm.write(f"  Total cost: ${total_cost:.4f}")

            if chunk_index < 0 or chunk_index >= len(chunks):
                continue

            row = {
                "title": page["title"],
                "chunk_index": chunk_index,
                "question": question,
                "answer": answer,
                "chunk_text": chunks[chunk_index],
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
