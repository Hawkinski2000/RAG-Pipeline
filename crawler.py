import wikipediaapi
from tqdm import tqdm


wiki = wikipediaapi.Wikipedia(
    user_agent="rag-pipeline (hawkinski2019@gmail.com)", language="en"
)


def crawl(seed, max_articles=200):
    seed_page = wiki.page(seed)
    links = list(seed_page.links.keys())[:max_articles]
    corpus = [{"title": seed, "text": seed_page.text}]

    with tqdm(links, desc="Crawling Wikipedia", unit="articles") as pbar:
        for title in pbar:
            page = wiki.page(title)
            if not page.exists():
                continue
            corpus.append({"title": title, "text": page.text})

    return corpus


corpus = crawl("Large_language_model")
print(corpus[0])
