"""
Databricks LLM client for API calls.
"""

import os
import streamlit as st
from openai import OpenAI
from typing import List, Dict, Any, Optional
from ..config import logger, USE_DATABRICKS_LLM

# Databricks endpoint URL
DATABRICKS_BASE_URL = "https://adb-3858882779799477.17.azuredatabricks.net/serving-endpoints"
DATABRICKS_LLM_MODEL = "databricks-llama-4-maverick"


@st.cache_resource
def get_databricks_llm_client():
    """
    Creates and caches an OpenAI client configured for Databricks LLM.

    Returns:
        OpenAI client or None if token is not available
    """
    try:
        # Check if Databricks LLM is enabled
        if not USE_DATABRICKS_LLM:
            logger.info("Databricks LLM is disabled in configuration")
            return None

        # Get token from environment variables only
        databricks_token = os.environ.get("DATABRICKS_API_KEY")

        if not databricks_token:
            logger.error("DATABRICKS_API_KEY not found in environment variables")
            st.error("Databricks API token not found. Please add DATABRICKS_API_KEY to your .env file")
            return None

        # Create OpenAI client with Databricks configuration
        client = OpenAI(
            api_key=databricks_token,
            base_url=DATABRICKS_BASE_URL
        )
        logger.info("Databricks LLM OpenAI client created successfully")
        return client
    except Exception as e:
        logger.error(f"Error creating Databricks LLM client: {e}")
        st.error(f"Failed to initialize Databricks LLM client: {e}")
        return None


class DatabricksLLMClient:
    """
    Client for interacting with Databricks LLM API.
    """

    def __init__(self):
        self.client = get_databricks_llm_client()
        self.model_name = DATABRICKS_LLM_MODEL

    def get_completion(self, messages: List[Dict[str, str]], max_tokens: int = 8192) -> Optional[str]:
        """
        Get completion from Databricks LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text or None if there was an error
        """
        if self.client is None:
            logger.error("Cannot get completion: Databricks LLM client not initialized")
            return None

        try:
            # Convert Anthropic-style messages to OpenAI format if needed
            openai_messages = []
            for msg in messages:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")

                # Map Anthropic roles to OpenAI roles
                if role == "system":
                    openai_messages.append({"role": "system", "content": content})
                elif role == "user":
                    openai_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    openai_messages.append({"role": "assistant", "content": content})
                else:
                    # Default to user for unknown roles
                    openai_messages.append({"role": "user", "content": content})

            # Make the API call
            response = self.client.chat.completions.create(
                messages=openai_messages,
                model=self.model_name,
                max_tokens=max_tokens
            )

            # Extract the generated text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.warning("Empty response from Databricks LLM API")
                return None

        except Exception as e:
            logger.error(f"Error getting completion from Databricks LLM: {e}")
            return None

    async def get_completion_async(self, messages: List[Dict[str, str]], max_tokens: int = 8192) -> Optional[str]:
        """
        Async wrapper for get_completion.
        This is a simple wrapper that calls the synchronous method, as the OpenAI client doesn't have async methods.
        For true async, you would need to run this in a thread pool.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text or None if there was an error
        """
        return self.get_completion(messages, max_tokens)


@st.cache_resource
def get_databricks_llm():
    """
    Creates and caches a DatabricksLLMClient instance.

    Returns:
        DatabricksLLMClient instance
    """
    try:
        logger.info(f"Initializing Databricks LLM client with model: {DATABRICKS_LLM_MODEL}")
        client = DatabricksLLMClient()

        # Test the client with a simple prompt
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, are you working?"}
        ]
        test_response = client.get_completion(test_messages, max_tokens=10)

        if test_response:
            logger.info(f"Databricks LLM client initialized successfully. Test response: {test_response[:20]}...")
            return client
        else:
            logger.error("Databricks LLM client test failed: No response")
            return None
    except Exception as e:
        logger.error(f"Error initializing Databricks LLM client: {e}")
        return None
