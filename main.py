from tqdm import tqdm
from crawler import crawl
from chunker import split_text
from embedder import get_embedding


def main():
    corpus = crawl(seed="Large_language_model", max_articles=1)

    embeddings = []
    for page in corpus:
        page_chunks = split_text(page["text"])
        page_embeddings = []
        for chunk in tqdm(
            page_chunks,
            desc=f'Generating embeddings for "{page["title"]}"',
            unit="chunk",
        ):
            embedding = get_embedding(chunk)
            page_embeddings.append(embedding)
        embeddings.append(page_embeddings)


if __name__ == "__main__":
    main()
