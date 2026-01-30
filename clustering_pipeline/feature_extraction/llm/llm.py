from abc import ABC, abstractmethod


class InstructLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, image_path: str) -> str:
        ...