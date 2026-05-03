def reciprocal_rank_fusion(*ranked_lists, k=60):
    scores = {}
    items = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank + 1)
            items[item_id] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [{"rrf_score": scores[i], **items[i]} for i in sorted_ids]
