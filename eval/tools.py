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

compute_faithfulness_tool = {
    "type": "function",
    "name": "compute_faithfulness",
    "description": "Evaluate whether an answer is faithful to the provided source chunks. Returns a faithfulness assessment with a score and explanation.",
    "parameters": {
        "type": "object",
        "properties": {
            "faithful": {
                "type": "boolean",
                "description": "Whether the answer is faithful to the source chunks",
            },
            "score": {
                "type": "number",
                "description": "Faithfulness score between 0.0 and 1.0",
            },
            "explanation": {
                "type": "string",
                "description": "Explanation of the faithfulness assessment",
            },
        },
        "required": ["faithful", "score", "explanation"],
        "additionalProperties": False,
    },
    "strict": True,
}
