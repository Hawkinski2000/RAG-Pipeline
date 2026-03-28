from tqdm import tqdm
from crawler import crawl
from chunker import split_text
from embedder import get_embeddings_batch


def main():
    corpus = crawl(seed="Large_language_model", max_articles=20)

    embeddings = []
    pbar = tqdm(corpus, desc="Generating embeddings", unit="page")
    for page in pbar:
        page_chunks = split_text(page["text"])
        page_embeddings = get_embeddings_batch(page_chunks)
        embeddings.append(page_embeddings)


if __name__ == "__main__":
    main()
