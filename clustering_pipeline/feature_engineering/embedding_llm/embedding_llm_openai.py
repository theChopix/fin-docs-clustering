import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from .embedding_llm import EmbeddingLLM

load_dotenv()

class OpenAIEmbeddingLLM(EmbeddingLLM):
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("EMBEDDING_MODEL"),
        )

    def generate_embedding(self, text: str) -> list[float]:
        embedding = self.embedding_model.embed_query(text)
        return embedding