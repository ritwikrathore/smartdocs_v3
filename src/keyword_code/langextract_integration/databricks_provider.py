"""
Custom Databricks provider for LangExtract.
This provider allows LangExtract to work with Databricks LLM endpoints using OpenAI-style calls.
"""

import os
import langextract as lx
from openai import OpenAI
from typing import List, Dict, Any, Optional
from ..config import logger


@lx.providers.registry.register(r'^databricks-', priority=10)
class DatabricksProviderLanguageModel(lx.inference.BaseLanguageModel):
    """LangExtract provider for Databricks LLM endpoints."""

    def __init__(self, model_id: str, api_key: str = None, base_url: str = None, **kwargs):
        """Initialize the Databricks provider."""
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('DATABRICKS_API_KEY')
        self.base_url = base_url or "https://adb-3858882779799477.17.azuredatabricks.net/serving-endpoints"
        
        if not self.api_key:
            raise ValueError("DATABRICKS_API_KEY environment variable is required")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"Initialized Databricks provider for model: {model_id}")

    @classmethod
    def get_schema_class(cls):
        """Tell LangExtract about our schema support."""
        from .databricks_schema import DatabricksProviderSchema
        return DatabricksProviderSchema

    def infer(self, batch_prompts, **kwargs):
        """Run inference on a batch of prompts."""
        for prompt in batch_prompts:
            try:
                # Prepare messages for the API call
                messages = [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that extracts structured information from documents. Provide accurate, factual information based only on the content provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
                # Make the API call
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_id,
                    max_tokens=kwargs.get('max_tokens', 5000),
                    temperature=kwargs.get('temperature', 0.1)
                )
                
                result = chat_completion.choices[0].message.content
                yield [lx.inference.ScoredOutput(score=1.0, output=result)]
                
            except Exception as e:
                logger.error(f"Databricks API error: {e}")
                raise lx.core.exceptions.InferenceRuntimeError(f"Databricks API error: {e}") from e
