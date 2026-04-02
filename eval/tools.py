generate_qa_pair_tool = {
    "type": "function",
    "name": "generate_qa_pair",
    "description": "Generate a question-answer pair grounded in one or more chunks from a page.",
    "parameters": {
        "type": "object",
        "properties": {
            "chunk_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Indices of chunks used to answer the question",
            },
            "question": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["chunk_indices", "question", "answer"],
        "additionalProperties": False,
    },
    "strict": True,
}
