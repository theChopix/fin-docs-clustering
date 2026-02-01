import json
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Type
from .llm.llm import InstructLLM

from .llm.llm_openai import OpenAIInstructLLM
from .models import ExtractedFeatures


def extract_features_from_image(image_path: str,
                                llm: InstructLLM = OpenAIInstructLLM(),
                                prompt_path: Path = Path(__file__).parent / Path('prompt/system_prompt.txt'),
                                response_model: Type[BaseModel] = ExtractedFeatures,
                                max_retries: int = 3
                                ) -> dict:  
    """
    Extract features from an image.

    Args:
        image_path (str): The path to the input image.
        llm (InstructLLM): The language model to use for feature extraction.
        prompt_path (Path): The path to the prompt file.
        response_model (Type[BaseModel]): The Pydantic model to validate the response.
        max_retries (int): The maximum number of retries for the LLM response.
    Returns:
        dict: A dictionary containing extracted features.
    """
    with open(prompt_path, 'r') as file:
        prompt = file.read()

    def is_valid(data_: dict) -> bool:
        """Validate the response data against the Pydantic model."""
        try:
            response_model(**data_)
            return True
        except ValidationError:
            return False

    # generate the response using the LLM client with retries
    data = {}
    for _ in range(max_retries + 1):

        response = llm.generate_response(prompt, image_path)

        try:
            data = json.loads(response)
            # if json is valid and matches the model, break the loop
            if is_valid(data): 
                break
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}
        
    return data
