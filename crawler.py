import os
import json
from datetime import datetime, timezone
import wikipediaapi
from tqdm import tqdm


MAX_LINKS = 200


def get_next_corpus_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpora_dir = os.path.join(script_dir, "corpora")
    os.makedirs(corpora_dir, exist_ok=True)

    existing_dirs = [
        dir_name
        for dir_name in os.listdir(corpora_dir)
        if dir_name.startswith("wiki_corpus_v")
    ]

    if not existing_dirs:
        version = 1
    else:
        versions = [
            int(dir_name.split("_v")[-1])
            for dir_name in existing_dirs
            if dir_name.split("_v")[-1].isdigit()
        ]
        version = max(versions) + 1

    corpus_name = f"wiki_corpus_v{version}"
    corpus_path = os.path.join(corpora_dir, corpus_name)

    os.makedirs(corpus_path, exist_ok=True)

    return corpus_path, corpus_name


def crawl(seed, max_links):
    corpus_path, corpus_name = get_next_corpus_path()
    data_path = os.path.join(corpus_path, "data.jsonl")
    meta_path = os.path.join(corpus_path, "meta.json")

    wiki = wikipediaapi.Wikipedia(
        user_agent="rag-pipeline (hawkinski2019@gmail.com)", language="en"
    )

    seed_page = wiki.page(seed)
    links = list(seed_page.links.keys())[: max_links - 1]

    with open(data_path, "w", encoding="utf-8") as f:
        seed_page_row = {
            "title": seed,
            "text": seed_page.text,
            "page_id": seed_page.pageid,
            "url": seed_page.canonicalurl,
            "page_length": len(seed_page.text),
        }
        f.write(json.dumps(seed_page_row, ensure_ascii=False) + "\n")
        total_pages = 1
        total_page_length = len(seed_page.text)

        for title in tqdm(links, desc="Crawling Wikipedia", unit="links"):
            page = wiki.page(title)
            if not page.exists():
                continue
            page_row = {
                "title": title,
                "text": page.text,
                "page_id": page.pageid,
                "url": page.canonicalurl,
                "page_length": len(page.text),
            }
            f.write(json.dumps(page_row, ensure_ascii=False) + "\n")
            total_pages += 1
            total_page_length += len(page.text)

    meta = {
        "name": corpus_name,
        "seed": seed,
        "num_pages": total_pages,
        "avg_page_length": round(total_page_length / total_pages, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Corpus saved to: {corpus_path}")


if __name__ == "__main__":
    crawl(seed="Large_language_model", max_links=MAX_LINKS)
