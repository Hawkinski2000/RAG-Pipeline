import re, pickle, pathlib
from rank_bm25 import BM25Okapi

INDEX_PATH = pathlib.Path("bm25_index.pkl")


def _tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        corpus = [_tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def load(self):
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def query(self, query, top_k):
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [
            {
                "id": self.chunks[i]["id"],
                "text": self.chunks[i]["text"],
                "title": self.chunks[i]["title"],
                "chunk_index": self.chunks[i]["chunk_index"],
            }
            for i in top_indices
        ]
