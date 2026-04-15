generate_chunk_context_tool = {
    "type": "function",
    "name": "generate_chunk_context",
    "description": "Generate concise contextual descriptions for multiple chunks from a single document page.",
    "parameters": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "description": "Context results for each chunk",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_index": {
                            "type": "integer",
                            "description": "Index of the chunk",
                        },
                        "context": {
                            "type": "string",
                            "description": "Short contextual sentence situating the chunk in the document",
                        },
                    },
                    "required": ["chunk_index", "context"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    },
    "strict": True,
}
