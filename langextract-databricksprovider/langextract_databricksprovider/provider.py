"""Provider implementation for DatabricksProvider."""

import os
import langextract as lx
from openai import OpenAI
from langextract_databricksprovider.schema import DatabricksProviderSchema


@lx.providers.registry.register(r'^databricks-', priority=10)
class DatabricksProviderLanguageModel(lx.inference.BaseLanguageModel):
    """LangExtract provider for Databricks LLM endpoints.

    This provider handles model IDs matching: ['^databricks-']
    """

    def __init__(self, model_id: str, api_key: str = None, base_url: str = None, **kwargs):
        """Initialize the Databricks provider.

        Args:
            model_id: The model identifier.
            api_key: API key for authentication.
            base_url: Base URL for the Databricks endpoint.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('DATABRICKS_API_KEY')
        self.base_url = base_url or "https://adb-3858882779799477.17.azuredatabricks.net/serving-endpoints"
        self.response_schema = kwargs.get('response_schema')
        self.structured_output = kwargs.get('structured_output', False)

        if not self.api_key:
            raise ValueError("DATABRICKS_API_KEY environment variable is required")

        # Initialize OpenAI client with Databricks configuration
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self._extra_kwargs = kwargs

    @classmethod
    def get_schema_class(cls):
        """Tell LangExtract about our schema support."""
        from langextract_databricksprovider.schema import DatabricksProviderSchema
        return DatabricksProviderSchema

    def apply_schema(self, schema_instance):
        """Apply or clear schema configuration."""
        super().apply_schema(schema_instance)
        if schema_instance:
            config = schema_instance.to_provider_config()
            self.response_schema = config.get('response_schema')
            self.structured_output = config.get('structured_output', False)
        else:
            self.response_schema = None
            self.structured_output = False

    def infer(self, batch_prompts, **kwargs):
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional inference parameters.

        Yields:
            Lists of ScoredOutput objects, one per prompt.
        """
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
                # Return an error result instead of raising
                error_result = f"Error calling Databricks API: {e}"
                yield [lx.inference.ScoredOutput(score=0.0, output=error_result)]
