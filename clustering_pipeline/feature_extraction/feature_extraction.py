import json
from pathlib import Path
from .llm.llm import InstructLLM

from .llm.llm_openai import OpenAIInstructLLM


def extract_features_from_image(image_path: str,
                                llm: InstructLLM = OpenAIInstructLLM(),
                                prompt_path: Path = Path(__file__).parent / Path('prompt/system_prompt.txt')
                                ) -> dict:  
    """
    Extract features from an image.

    Args:
        image_path (str): The path to the input image.
        llm (InstructLLM): The language model to use for feature extraction.
        prompt_path (Path): The path to the prompt file.
    Returns:
        dict: A dictionary containing extracted features.
    """
    with open(prompt_path, 'r') as file:
        prompt = file.read()

    extracted_features_str = llm.generate_response(prompt, image_path)
    extracted_features = json.loads(extracted_features_str)

    return extracted_features
