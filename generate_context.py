import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from crawler import crawl
from chunker import split_text


MAX_LINKS = 2
MODEL = "gpt-5.4-nano"
PRICING = {"input": 0.20, "output": 1.25}
MAX_OUTPUT_TOKENS = 500

load_dotenv()


def build_context_prompt(document, chunk):
    return f"""
<document> 
{document} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
"""


def generate_context(seed, max_links):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    context_dir = os.path.join(script_dir, "context")
    os.makedirs(context_dir, exist_ok=True)

    data_path = os.path.join(context_dir, "data.jsonl")
    meta_path = os.path.join(context_dir, "meta.json")

    openai_client = OpenAI()

    corpus = crawl(seed=seed, max_links=max_links)

    total_pages = 0
    total_chunks = 0
    total_cost = 0.0

    with open(data_path, "a", encoding="utf-8") as f:
        for page in tqdm(corpus, desc="Generating context for page", unit="pages"):
            chunks = split_text(page["text"])

            for chunk in tqdm(
                chunks, desc="Generating context for chunk", unit="chunks"
            ):
                prompt = build_context_prompt(page["text"], chunk)

                response = openai_client.responses.create(
                    model=MODEL,
                    input=prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )

                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                chunk_cost = (
                    input_tokens * PRICING["input"] + output_tokens * PRICING["output"]
                ) / 1000000
                total_cost += chunk_cost

                row = {
                    "title": page["title"],
                    "chunk": chunk,
                    "context": response.output_text,
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_chunks += 1

            tqdm.write(f"  Total cost: ${total_cost:.4f}")
            total_pages += 1

    meta = {
        "seed": seed,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "avg_chunks_per_page": round(total_chunks / total_pages, 1),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Context saved to: {data_path}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    generate_context(seed="Large_language_model", max_links=MAX_LINKS)
