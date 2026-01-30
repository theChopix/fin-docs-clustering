import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import base64

from .llm import InstructLLM

load_dotenv()

class OpenAIInstructLLM(InstructLLM):
    def __init__(self, temperature: float = 0.0):
        self.model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                                model=os.getenv("CHAT_MODEL"),
                                max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
                                temperature=temperature,)
        

    def generate_response(self, prompt: str, image_path: str) -> str:

        def encode_image(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        b64 = encode_image(image_path)

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    },
                },
            ]
        )

        response = self.model.invoke([message])
        return response.content