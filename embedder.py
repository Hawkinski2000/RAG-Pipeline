import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)


def get_embeddings_batch(texts):
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = [item.embedding for item in response.data]
    return embeddings
