from zeroentropy import ZeroEntropy


zclient = ZeroEntropy()


def rerank_chunks(query, chunks, top_n=None):
    documents = [chunk["text"] for chunk in chunks]

    response = zclient.models.rerank(
        model="zerank-2", query=query, documents=documents, top_n=top_n
    )

    reranked_chunks = [chunks[result.index] for result in response.results]

    return reranked_chunks
