import logging
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import openai
import os
from ..config.settings import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OPENROUTER_TOKEN,
    MAX_MEMORY_TOKENS
)


class LLMProvider:
    def __init__(self, provider: str = "ollama", model_name: str = "llama2"):
        self.provider = provider.lower()
        self.model_name = model_name
        self.parser = JsonOutputParser()

        if self.provider == "ollama":
            self.llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.0
            )
        elif self.provider == "openrouter":
            openai.api_key = OPENROUTER_TOKEN
            openai.api_base = "https://openrouter.ai/api/v1"
            self.llm = None  # Will use OpenAI client directly
        elif self.provider == "openai":
            openai.api_key = os.getenv("API_KEY_OPENAI")
            self.llm = None  # Will use OpenAI client directly
        elif self.provider == "gemini":
            from google import genai
            genai.configure(api_key=os.getenv("API_KEY_GOOGLE"))
            self.llm = None  # Will use Gemini client directly

    def generate_response(
        self,
        prompt: str,
        task_name: str,
        task_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates a response from the LLM using the provided prompt and task definition.
        """
        try:
            # Create the prompt template
            template = PromptTemplate(
                template=task_definition["prompt_template"],
                input_variables=["input"]
            )

            # Format the prompt
            formatted_prompt = template.format(input=prompt)

            # Generate response based on provider
            if self.provider == "ollama":
                response = self.llm.invoke(formatted_prompt)
            elif self.provider in ["openai", "openrouter"]:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": formatted_prompt}]
                )
                response = response.choices[0].message.content
            elif self.provider == "gemini":
                from google import genai
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(formatted_prompt)
                response = response.text

            # Parse the response
            parsed_response = self.parser.parse(response)

            # Add task-specific metadata
            parsed_response["task_name"] = task_name
            parsed_response["prompt"] = formatted_prompt

            return parsed_response

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return {
                "error": str(e),
                "task_name": task_name,
                "prompt": prompt
            }

    def validate_response(
        self,
        response: Dict[str, Any],
        task_definition: Dict[str, Any]
    ) -> bool:
        """
        Validates the response against the task's expected format and rules.
        """
        try:
            # Check if response has required fields
            required_fields = task_definition.get("required_fields", [])
            for field in required_fields:
                if field not in response:
                    return False

            # Task-specific validation
            if task_definition["name"] == "sentiment_analysis":
                return response["sentiment"] in ["positive", "negative", "neutral"]
            elif task_definition["name"] == "medical_info":
                return all(key in response for key in ["condition", "symptoms", "treatment"])
            elif task_definition["name"] == "review_classification":
                return response["category"] in ["positive", "negative", "neutral"]
            elif task_definition["name"] == "industrial_machinery":
                return all(key in response for key in ["machine_type", "maintenance_needs", "operational_status"])
            elif task_definition["name"] == "fake_news":
                return response["is_fake"] in [True, False]

            return True

        except Exception as e:
            logging.error(f"Error validating response: {e}")
            return False

    def get_feedback_prompt(
        self,
        task_definition: Dict[str, Any],
        response: Dict[str, Any]
    ) -> str:
        """
        Generates a feedback prompt based on the task definition and response.
        """
        return task_definition["feedback_prompt"].format(
            input=response.get("input", ""),
            response=response
        )
