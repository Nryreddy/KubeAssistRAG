"""
LLM Service.

Provides wrapper functions for executing ChatCompletion calls against OpenAI models.
Supports both standard text generation and JSON-mode generation.
"""
from openai import OpenAI

from app.config import settings


openai_client = OpenAI(api_key=settings.openai_api_key)


def generate(system_prompt: str, user_message: str, model: str | None = None, temperature: float = 0.0) -> dict:
    """
    Generate a text response from the LLM.
    
    Args:
        system_prompt: High-level instructions for the model.
        user_message: The specific user prompt.
        model: Optional model override.
        temperature: Sampling temperature.
        
    Returns:
        A dictionary containing the generated 'text' and token 'usage'.
    """
    if model is None:
        model = settings.llm_model_answer

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )

    text = response.choices[0].message.content or ""

    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }

    return {"text": text, "usage": usage}

def generate_with_json(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """
    Generate a JSON object response from the LLM.
    
    Enforces `response_format={"type": "json_object"}`. The system prompt
    should instruct the model to output JSON.
    
    Args:
        system_prompt: High-level instructions for the model.
        user_message: The specific user prompt.
        model: Optional model override.
        temperature: Sampling temperature.
        
    Returns:
        A dictionary containing the generated 'text' (JSON string) and token 'usage'.
    """
    if model is None:
        model = settings.llm_model_grader

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return {"text": text, "usage": usage}