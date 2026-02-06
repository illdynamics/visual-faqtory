# vfaq/llm_utils.py
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MockLLMClient:
    """A mock LLM client for testing and when no real LLM is configured."""
    def generate_content(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        logger.info("Mock LLM called. Returning dummy response.")
        # Return a valid JSON structure for InspeQtor to parse
        return """
{
    "quality_assessment": "The visuals are coherent and vibrant.",
    "evolution_suggestion": "increase saturation slightly and add a subtle chromatic aberration effect",
    "color_evolution": "shift towards brighter, more energetic hues",
    "energy_evolution": "increase",
    "innovative_idea": "introduce visual echoes that subtly trail moving elements",
    "reasoning": "This evolution will make the visuals more dynamic and engaging without drastic changes, enhancing the 'innovative mode' feel."
}
"""

def create_llm_client(llm_config: Dict[str, Any]) -> Any:
    """
    Creates and returns an LLM client based on the provided configuration.
    """
    provider = llm_config.get('provider', 'mock').lower()
    model = llm_config.get('model')
    api_key_env = llm_config.get('api_key_env')

    api_key = os.getenv(api_key_env) if api_key_env else None

    if provider == 'openai':
        try:
            from openai import OpenAI
            if not api_key:
                logger.warning(f"OpenAI API key not found. Please set the '{api_key_env}' environment variable. Using Mock LLM.")
                return MockLLMClient()
            logger.info(f"Initializing OpenAI client with model: {model}")
            return OpenAI(api_key=api_key)
        except ImportError:
            logger.error("OpenAI library not found. Please install it (`pip install openai`). Falling back to Mock LLM.")
            return MockLLMClient()
    elif provider == 'google':
        try:
            import google.generativeai as genai
            if not api_key:
                logger.warning(f"Google Gemini API key not found. Please set the '{api_key_env}' environment variable. Using Mock LLM.")
                return MockLLMClient()
            genai.configure(api_key=api_key)
            logger.info(f"Initializing Google Gemini client with model: {model}")
            return genai.GenerativeModel(model)
        except ImportError:
            logger.error("Google Generative AI library not found. Please install it (`pip install google-generativeai`). Falling back to Mock LLM.")
            return MockLLMClient()
    elif provider == 'mock':
        logger.info("Initializing Mock LLM client.")
        return MockLLMClient()
    else:
        logger.warning(f"Unknown LLM provider '{provider}'. Falling back to Mock LLM.")
        return MockLLMClient()

def call_llm(llm_client: Any, system_prompt: str, user_prompt: str, config: Dict[str, Any]) -> str:
    """
    Calls the LLM client to generate a response.
    """
    llm_config = config.get('llm', {})
    provider = llm_config.get('provider', 'mock').lower()
    model = llm_config.get('model')

    if isinstance(llm_client, MockLLMClient):
        return llm_client.generate_content(user_prompt, system_prompt)

    try:
        if provider == 'openai':
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        elif provider == 'google':
            # Google Gemini's API for system instructions and JSON mode
            # This part needs careful handling as Gemini's API can vary.
            # Assuming `generate_content` is the method.
            # For JSON output with Gemini, often you instruct it in the prompt itself.
            response = llm_client.generate_content(
                contents=[
                    {"role": "user", "parts": [system_prompt, user_prompt]}
                ]
            )
            # This is a simplification; a more robust solution would
            # handle response parsing and error checking.
            return response.text
        else:
            logger.warning(f"LLM provider '{provider}' not fully supported in call_llm. Falling back to Mock LLM behavior.")
            return MockLLMClient().generate_content(user_prompt, system_prompt)
    except Exception as e:
        logger.error(f"Error calling LLM provider '{provider}': {e}. Falling back to Mock LLM behavior.", exc_info=True)
        return MockLLMClient().generate_content(user_prompt, system_prompt)
