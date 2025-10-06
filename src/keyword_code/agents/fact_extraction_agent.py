"""
Pydantic-AI agent for intelligent fact extraction from analysis text.
Replaces LangExtract with custom LLM-based extraction that dynamically identifies fact types.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
import re

from ..config import logger
from ..ai.databricks_llm import DatabricksLLMClient, DATABRICKS_BASE_URL, DATABRICKS_LLM_MODEL


def strip_markdown_json(response: str) -> str:
    """
    Strip markdown code block formatting from JSON responses.

    LLMs often wrap JSON in ```json ... ``` blocks, which breaks JSON parsing.
    This function removes those wrappers.

    Args:
        response: Raw response from LLM

    Returns:
        Cleaned JSON string
    """
    response_clean = response.strip()

    # Remove opening markdown fence
    if response_clean.startswith("```json"):
        response_clean = response_clean[7:]
    elif response_clean.startswith("```"):
        response_clean = response_clean[3:]

    # Remove closing markdown fence
    if response_clean.endswith("```"):
        response_clean = response_clean[:-3]

    return response_clean.strip()


class FactType(str, Enum):
    """Enumeration of supported fact types."""
    DEFINITION = "definition"
    PERCENTAGE = "percentage"
    AMOUNT = "amount"
    CURRENCY_AMOUNT = "currency_amount"
    DATE = "date"
    ENTITY = "entity"
    RATE = "rate"
    DURATION = "duration"
    BOOLEAN = "boolean"
    LIST = "list"
    OTHER = "other"


class FactTypeAnalysis(BaseModel):
    """Analysis of expected fact types from a query."""
    query: str = Field(description="The original query/sub-prompt")
    expected_fact_types: List[FactType] = Field(
        description="List of fact types expected to be found in the analysis"
    )
    reasoning: str = Field(description="Explanation of why these fact types are expected")
    confidence: float = Field(
        description="Confidence score (0-1) in the fact type identification"
    )
    extraction_hints: List[str] = Field(
        default_factory=list,
        description="Hints for the extraction agent (e.g., 'Look for multiple loan amounts')"
    )

    @field_validator('confidence')
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        """Ensure confidence is within valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class ExtractedFact(BaseModel):
    """A single extracted fact with flexible structure."""
    fact_name: str = Field(description="Name/label of the fact (e.g., 'Business Day', 'Loan Amount A')")
    fact_value: str = Field(description="The extracted value or definition")
    fact_type: FactType = Field(description="Type of the fact")
    confidence: float = Field(
        description="Confidence in the extraction (0-1)",
        default=1.0
    )
    source_text: Optional[str] = Field(
        default=None,
        description="The exact text from which this fact was extracted"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., currency, unit, format)"
    )

    @field_validator('fact_value')
    @classmethod
    def validate_fact_value(cls, v: str, info) -> str:
        """Ensure fact value is not empty and validate based on fact type."""
        if not v or not v.strip():
            raise ValueError("Fact value cannot be empty")

        # Get fact_type from the model if available
        fact_type = info.data.get('fact_type') if hasattr(info, 'data') else None

        # Type-specific validation
        if fact_type == FactType.PERCENTAGE:
            # Check if value contains a percentage or numeric value
            if not any(char.isdigit() for char in v):
                raise ValueError(f"Percentage value must contain numeric data: {v}")

        elif fact_type == FactType.CURRENCY_AMOUNT or fact_type == FactType.AMOUNT:
            # Check if value contains numeric data
            if not any(char.isdigit() for char in v):
                raise ValueError(f"Amount value must contain numeric data: {v}")

        elif fact_type == FactType.DATE:
            # Basic date validation - should contain numbers
            if not any(char.isdigit() for char in v):
                raise ValueError(f"Date value must contain numeric data: {v}")

        return v.strip()

    @field_validator('fact_name')
    @classmethod
    def validate_fact_name(cls, v: str) -> str:
        """Ensure fact name is not empty and is descriptive."""
        if not v or not v.strip():
            raise ValueError("Fact name cannot be empty")

        # Ensure fact name is at least 2 characters
        if len(v.strip()) < 2:
            raise ValueError(f"Fact name too short: {v}")

        return v.strip()

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class FactExtractionResult(BaseModel):
    """Result of fact extraction from analysis text."""
    query: str = Field(description="The original query/sub-prompt")
    analysis_text: str = Field(description="The analysis text that was processed")
    extracted_facts: List[ExtractedFact] = Field(
        default_factory=list,
        description="List of extracted facts"
    )
    fact_types_found: List[FactType] = Field(
        default_factory=list,
        description="Types of facts that were found"
    )
    extraction_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the extraction process"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during extraction"
    )


class FactTypeIdentificationAgent:
    """
    Pydantic-AI agent for identifying expected fact types from queries.
    """
    
    def __init__(self, databricks_client: DatabricksLLMClient):
        self.databricks_client = databricks_client
        self.model = self._create_databricks_model()
        
        # Create the agent for fact type identification
        self.agent = Agent(
            model=self.model,
            output_type=FactTypeAnalysis,
            system_prompt="""You are an expert at analyzing queries to identify what types of facts should be extracted.

Your job is to analyze a user's query or sub-prompt and determine what types of factual information they are looking for.

Fact Types:
- DEFINITION: Explanations, meanings, or definitions (e.g., "What is a business day?")
- PERCENTAGE: Percentage values (e.g., "What is the interest rate?")
- AMOUNT: Numeric amounts without currency (e.g., "How many days?")
- CURRENCY_AMOUNT: Monetary amounts with currency (e.g., "What is the loan amount?")
- DATE: Dates or time periods (e.g., "When is the maturity date?")
- ENTITY: Names of people, organizations, or entities (e.g., "Who is the borrower?")
- RATE: Rates or ratios (e.g., "What is the exchange rate?")
- DURATION: Time durations (e.g., "What is the loan term?")
- BOOLEAN: Yes/no or true/false values (e.g., "Is it secured?")
- LIST: Lists of items (e.g., "What are the conditions?")
- OTHER: Any other type of fact

Guidelines:
1. A query may expect multiple fact types
2. If a query asks about multiple items (e.g., "loan amounts for A and B"), note this in extraction_hints
3. Be specific in your reasoning
4. Provide high confidence (0.8-1.0) for clear queries, lower (0.5-0.7) for ambiguous ones

Analyze the query and identify the expected fact types."""
        )
    
    def _create_databricks_model(self):
        """Create an OpenAIChatModel configured for Databricks."""
        try:
            api_key = os.environ.get("DATABRICKS_API_KEY")
            if not api_key:
                raise RuntimeError("DATABRICKS_API_KEY is not set in environment variables")

            provider = OpenAIProvider(api_key=api_key, base_url=DATABRICKS_BASE_URL)
            # Use OpenAIChatModel without structured output constraints
            # Databricks doesn't support all JSON schema features
            return OpenAIChatModel(DATABRICKS_LLM_MODEL, provider=provider)
        except Exception as e:
            logger.error(f"Error creating Databricks model: {e}")
            return None
    
    async def identify_fact_types(self, query: str, context: str = "") -> FactTypeAnalysis:
        """
        Identify expected fact types from a query.

        Args:
            query: The user's query or sub-prompt
            context: Additional context about the query

        Returns:
            FactTypeAnalysis with expected fact types and reasoning
        """
        try:
            prompt = f"""
            Analyze this query to identify what types of facts should be extracted.

            Query: "{query}"
            {f"Context: {context}" if context else ""}

            Respond with a JSON object containing:
            - query: the original query
            - expected_fact_types: list of fact type strings (definition, percentage, amount, currency_amount, date, entity, rate, duration, boolean, list, other)
            - reasoning: explanation of why these fact types are expected
            - confidence: confidence score between 0.0 and 1.0
            - extraction_hints: list of hints for extraction (e.g., "Look for multiple loan amounts")

            Example response:
            {{
                "query": "What is the interest rate?",
                "expected_fact_types": ["percentage", "rate"],
                "reasoning": "Query asks for interest rate which is typically expressed as a percentage",
                "confidence": 0.95,
                "extraction_hints": ["Look for percentage values", "Check for 'per annum' or similar terms"]
            }}

            Respond ONLY with the JSON object, no other text.
            """

            # Use Databricks client directly to avoid JSON schema issues
            messages = [
                {"role": "system", "content": "You are an expert at analyzing queries. Respond only with valid JSON. Do not wrap the JSON in markdown code blocks."},
                {"role": "user", "content": prompt}
            ]

            response = self.databricks_client.get_completion(messages, max_tokens=1000)

            if not response:
                raise ValueError("No response from LLM")

            # Strip markdown code blocks if present
            response_clean = strip_markdown_json(response)

            # Parse JSON response
            response_json = json.loads(response_clean)

            # Convert to FactTypeAnalysis
            return FactTypeAnalysis(
                query=response_json.get("query", query),
                expected_fact_types=[FactType(ft) for ft in response_json.get("expected_fact_types", ["definition"])],
                reasoning=response_json.get("reasoning", ""),
                confidence=float(response_json.get("confidence", 0.5)),
                extraction_hints=response_json.get("extraction_hints", [])
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in fact type identification: {e}")
            if 'response' in locals():
                logger.error(f"Original response: {response[:500]}...")
            if 'response_clean' in locals():
                logger.error(f"Cleaned response: {response_clean[:500]}...")
            # Return default analysis
            return FactTypeAnalysis(
                query=query,
                expected_fact_types=[FactType.DEFINITION],
                reasoning="Default to definition type due to JSON parsing error",
                confidence=0.3,
                extraction_hints=[]
            )
        except Exception as e:
            logger.error(f"Error in fact type identification: {e}", exc_info=True)
            # Return default analysis
            return FactTypeAnalysis(
                query=query,
                expected_fact_types=[FactType.DEFINITION],
                reasoning="Default to definition type due to analysis error",
                confidence=0.3,
                extraction_hints=[]
            )


class FactExtractionAgent:
    """
    Pydantic-AI agent for extracting facts from analysis text.
    """
    
    def __init__(self, databricks_client: DatabricksLLMClient):
        self.databricks_client = databricks_client
        self.model = self._create_databricks_model()
        
        # Create the agent for fact extraction
        self.agent = Agent(
            model=self.model,
            output_type=FactExtractionResult,
            system_prompt="""You are an expert at extracting structured facts from analysis text.

Your job is to extract specific facts from analysis text based on the expected fact types.

Guidelines:
1. Extract ONLY facts that are explicitly stated in the analysis text
2. Do NOT infer or add information that is not present
3. If multiple instances of the same fact type exist (e.g., "Loan Amount A" and "Loan Amount B"), extract them separately
4. Use clear, descriptive fact names (e.g., "Business Day Definition", "Interest Rate", "Loan Amount A")
5. For definitions, the fact_value should be the complete definition
6. For numeric values, include units in metadata (e.g., currency, percentage symbol)
7. Include source_text when possible to show where the fact was found
8. Set confidence based on how clearly the fact is stated (0.8-1.0 for clear, 0.5-0.7 for ambiguous)

Fact Type Specific Guidelines:
- DEFINITION: Extract the complete definition or explanation
- PERCENTAGE: Extract the numeric value and note "%" in metadata
- CURRENCY_AMOUNT: Extract amount and currency (e.g., "$500 million")
- DATE: Extract in the format found in the text
- ENTITY: Extract the full name of the entity
- RATE: Extract the rate value with units
- DURATION: Extract the time period with units
- BOOLEAN: Extract as "Yes"/"No" or "True"/"False"
- LIST: Extract as comma-separated items or numbered list

Be thorough and accurate in your extractions."""
        )
    
    def _create_databricks_model(self):
        """Create an OpenAIChatModel configured for Databricks."""
        try:
            api_key = os.environ.get("DATABRICKS_API_KEY")
            if not api_key:
                raise RuntimeError("DATABRICKS_API_KEY is not set in environment variables")
            
            provider = OpenAIProvider(api_key=api_key, base_url=DATABRICKS_BASE_URL)
            return OpenAIChatModel(DATABRICKS_LLM_MODEL, provider=provider)
        except Exception as e:
            logger.error(f"Error creating Databricks model: {e}")
            return None
    
    async def extract_facts(
        self,
        query: str,
        analysis_text: str,
        expected_fact_types: List[FactType],
        extraction_hints: List[str] = None,
        max_retries: int = 2
    ) -> FactExtractionResult:
        """
        Extract facts from analysis text with intelligent retry logic.

        Args:
            query: The original query/sub-prompt
            analysis_text: The analysis text to extract facts from
            expected_fact_types: List of expected fact types
            extraction_hints: Optional hints for extraction
            max_retries: Maximum number of retry attempts

        Returns:
            FactExtractionResult with extracted facts
        """
        attempt = 0
        last_error = None
        validation_errors = []

        while attempt <= max_retries:
            try:
                # Build the extraction prompt with adaptive instructions based on attempt
                fact_types_str = ", ".join([ft.value for ft in expected_fact_types])
                hints_str = "\n".join([f"- {hint}" for hint in (extraction_hints or [])])

                # Add retry-specific guidance
                retry_guidance = ""
                if attempt > 0:
                    retry_guidance = f"""

                    IMPORTANT - This is retry attempt {attempt + 1}:
                    - Previous attempt had issues: {last_error if last_error else 'No facts found'}
                    - Be more thorough and extract ALL relevant facts
                    - Ensure fact names are descriptive and unique
                    - Validate that numeric values contain actual numbers
                    - Double-check that all required fields are filled
                    """

                    if validation_errors:
                        retry_guidance += f"\n- Previous validation errors: {'; '.join(validation_errors[-3:])}"

                prompt = f"""
                Extract facts from the following analysis text.

                Original Query: "{query}"

                Expected Fact Types: {fact_types_str}

                {f"Extraction Hints:\n{hints_str}" if hints_str else ""}
                {retry_guidance}

                Analysis Text:
                {analysis_text}

                Extract all relevant facts based on the expected fact types. Be thorough and accurate.
                Ensure each fact has:
                - A clear, descriptive fact_name (e.g., "Interest Rate", "Loan Amount A")
                - A complete fact_value with the actual data
                - Appropriate fact_type matching the content
                - High confidence (0.8-1.0) for clearly stated facts

                Respond with a JSON object containing:
                - query: the original query
                - analysis_text: the analysis text (can be truncated)
                - extracted_facts: array of fact objects, each with:
                  * fact_name: string
                  * fact_value: string
                  * fact_type: one of ({fact_types_str})
                  * confidence: number between 0.0 and 1.0
                  * source_text: optional string
                  * metadata: optional object
                - fact_types_found: array of fact type strings
                - extraction_metadata: object with any metadata
                - errors: array of error strings (empty if no errors)

                Respond ONLY with the JSON object, no other text.
                """

                # Use Databricks client directly to avoid JSON schema issues
                messages = [
                    {"role": "system", "content": "You are an expert at extracting structured facts. Respond only with valid JSON. Do not wrap the JSON in markdown code blocks."},
                    {"role": "user", "content": prompt}
                ]

                response = self.databricks_client.get_completion(messages, max_tokens=4000)

                if not response:
                    raise ValueError("No response from LLM")

                # Strip markdown code blocks if present
                response_clean = strip_markdown_json(response)

                # Parse JSON response
                response_json = json.loads(response_clean)

                # Convert to FactExtractionResult
                extracted_facts = []
                for fact_data in response_json.get("extracted_facts", []):
                    try:
                        fact = ExtractedFact(
                            fact_name=fact_data.get("fact_name", ""),
                            fact_value=fact_data.get("fact_value", ""),
                            fact_type=FactType(fact_data.get("fact_type", "other")),
                            confidence=float(fact_data.get("confidence", 0.5)),
                            source_text=fact_data.get("source_text"),
                            metadata=fact_data.get("metadata", {})
                        )
                        extracted_facts.append(fact)
                    except Exception as fact_err:
                        logger.warning(f"Failed to parse fact: {fact_err}")
                        continue

                extraction_result = FactExtractionResult(
                    query=query,
                    analysis_text=analysis_text,
                    extracted_facts=extracted_facts,
                    fact_types_found=[FactType(ft) for ft in response_json.get("fact_types_found", [])],
                    extraction_metadata=response_json.get("extraction_metadata", {}),
                    errors=response_json.get("errors", [])
                )

                # Validate the result
                if not extraction_result.extracted_facts and attempt < max_retries:
                    logger.warning(f"No facts extracted on attempt {attempt + 1}, retrying with enhanced prompt...")
                    last_error = "No facts extracted from analysis text"
                    attempt += 1
                    continue

                # Additional validation: check for low-quality extractions
                if extraction_result.extracted_facts:
                    valid_facts = []
                    for fact in extraction_result.extracted_facts:
                        try:
                            # Validate fact quality
                            if len(fact.fact_name) < 2:
                                validation_errors.append(f"Fact name too short: {fact.fact_name}")
                                continue
                            if len(fact.fact_value) < 2:
                                validation_errors.append(f"Fact value too short: {fact.fact_value}")
                                continue

                            # Type-specific validation
                            if fact.fact_type in [FactType.PERCENTAGE, FactType.AMOUNT, FactType.CURRENCY_AMOUNT]:
                                if not any(char.isdigit() for char in fact.fact_value):
                                    validation_errors.append(f"Numeric fact missing numbers: {fact.fact_name}")
                                    continue

                            valid_facts.append(fact)
                        except Exception as val_err:
                            validation_errors.append(f"Validation error for {fact.fact_name}: {str(val_err)}")
                            continue

                    # If we filtered out facts and have retries left, try again
                    if len(valid_facts) < len(extraction_result.extracted_facts) and attempt < max_retries:
                        logger.warning(
                            f"Filtered {len(extraction_result.extracted_facts) - len(valid_facts)} invalid facts, "
                            f"retrying... (attempt {attempt + 1})"
                        )
                        last_error = f"Quality validation failed: {'; '.join(validation_errors[-3:])}"
                        attempt += 1
                        continue

                    extraction_result.extracted_facts = valid_facts

                # Add metadata
                extraction_result.extraction_metadata.update({
                    "model_used": DATABRICKS_LLM_MODEL,
                    "attempt": attempt + 1,
                    "expected_fact_types": [ft.value for ft in expected_fact_types],
                    "validation_errors": validation_errors if validation_errors else []
                })

                logger.info(
                    f"Successfully extracted {len(extraction_result.extracted_facts)} facts "
                    f"on attempt {attempt + 1}"
                )

                return extraction_result

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if 'response' in locals():
                    logger.error(f"Original response: {response[:500]}...")
                if 'response_clean' in locals():
                    logger.error(f"Cleaned response: {response_clean[:500]}...")
                last_error = f"JSON parsing error: {str(e)}"
                validation_errors.append(f"JSON parsing error: {str(e)}")
                attempt += 1

            except ValidationError as e:
                logger.error(f"Validation error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                validation_errors.append(str(e))
                attempt += 1

            except Exception as e:
                logger.error(f"Error in fact extraction on attempt {attempt + 1}: {e}", exc_info=True)
                last_error = str(e)
                attempt += 1

        # All retries failed, return empty result with error
        logger.error(f"Fact extraction failed after {max_retries + 1} attempts")
        return FactExtractionResult(
            query=query,
            analysis_text=analysis_text,
            extracted_facts=[],
            fact_types_found=[],
            extraction_metadata={
                "model_used": DATABRICKS_LLM_MODEL,
                "attempts": max_retries + 1,
                "expected_fact_types": [ft.value for ft in expected_fact_types],
                "validation_errors": validation_errors
            },
            errors=[
                f"Extraction failed after {max_retries + 1} attempts: {last_error}",
                *validation_errors[-5:]  # Include last 5 validation errors
            ]
        )

