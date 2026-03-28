import wikipediaapi
from tqdm import tqdm


wiki = wikipediaapi.Wikipedia(
    user_agent="rag-pipeline (hawkinski2019@gmail.com)", language="en"
)


def crawl(seed, max_links=200):
    seed_page = wiki.page(seed)
    links = list(seed_page.links.keys())[:max_links]
    corpus = [{"title": seed, "text": seed_page.text}]

    for title in tqdm(links, desc="Crawling Wikipedia", unit="links"):
        page = wiki.page(title)
        if not page.exists():
            continue
        corpus.append({"title": title, "text": page.text})

    return corpus
