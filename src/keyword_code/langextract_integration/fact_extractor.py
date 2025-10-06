"""
Fact extraction service using LangExtract.
This module processes LLM analysis responses to extract structured facts.
"""

import langextract as lx
import textwrap
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from ..config import logger


class FactExtractor:
    """Service for extracting structured facts from LLM analysis responses."""

    def __init__(self, model_id: str = "databricks-llama-4-maverick"):
        """
        Initialize the fact extractor.

        Args:
            model_id: The model ID to use for extraction (using Databricks as default)
        """
        self.model_id = model_id

        # Import the custom provider to ensure it's registered
        try:
            import langextract_databricksprovider
            logger.info("Successfully imported Databricks provider for LangExtract")
        except ImportError as e:
            logger.warning(f"Failed to import Databricks provider: {e}")
        except AttributeError as e:
            logger.warning(f"LangExtract providers not available: {e}")
        except Exception as e:
            logger.warning(f"Issue with LangExtract provider setup: {e}")

        # Check for Databricks API key only (workspace policy: Databricks-only)
        if not os.environ.get('DATABRICKS_API_KEY'):
            logger.warning("DATABRICKS_API_KEY not found in environment variables")

    def extract_facts_from_analysis(self, analysis_text: str, sub_prompt: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Extract structured facts from an analysis text using LangExtract.

        Args:
            analysis_text: The LLM analysis text to extract facts from
            sub_prompt: The original sub-prompt that generated this analysis
            context: Additional context about the analysis

        Returns:
            Dictionary containing extracted facts or None if extraction fails
        """
        try:
            # Define the extraction prompt
            prompt = textwrap.dedent("""
                Extract key facts and information from the provided analysis text.
                Focus on:
                - Specific data points (numbers, dates, amounts, percentages)
                - Key entities (names, organizations, locations)
                - Important relationships and connections
                - Factual statements and conclusions
                - Any quantitative or qualitative findings

                Extract only factual information that is explicitly stated in the analysis.
                Do not infer or add information that is not directly present.
            """)

            # Create examples to guide the extraction
            examples = [
                lx.data.ExampleData(
                    text=textwrap.dedent("""
                        The loan agreement specifies a total amount of $500 million with an interest rate of 3.5% per annum.
                        The maturity date is set for December 31, 2030. The borrower is ABC Corporation,
                        and the lender is XYZ Bank. The loan includes a refinancing component of $300 million
                        and a greenfield component of $200 million.
                    """),
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="financial_amount",
                            extraction_text="$500 million",
                            attributes={"type": "total_loan_amount", "currency": "USD"}
                        ),
                        lx.data.Extraction(
                            extraction_class="interest_rate",
                            extraction_text="3.5% per annum",
                            attributes={"type": "annual_rate", "value": "3.5"}
                        ),
                        lx.data.Extraction(
                            extraction_class="date",
                            extraction_text="December 31, 2030",
                            attributes={"type": "maturity_date", "format": "full_date"}
                        ),
                        lx.data.Extraction(
                            extraction_class="entity",
                            extraction_text="ABC Corporation",
                            attributes={"role": "borrower", "type": "corporation"}
                        ),
                        lx.data.Extraction(
                            extraction_class="entity",
                            extraction_text="XYZ Bank",
                            attributes={"role": "lender", "type": "bank"}
                        ),
                        lx.data.Extraction(
                            extraction_class="financial_amount",
                            extraction_text="$300 million",
                            attributes={"type": "refinancing_component", "currency": "USD"}
                        ),
                        lx.data.Extraction(
                            extraction_class="financial_amount",
                            extraction_text="$200 million",
                            attributes={"type": "greenfield_component", "currency": "USD"}
                        ),
                    ]
                )
            ]

            # Prepare the input text with context
            input_text = f"""
            Original Question: {sub_prompt}

            Analysis Text:
            {analysis_text}

            Additional Context: {context}
            """
            # Run the extraction

            logger.info(f"Running fact extraction for sub-prompt: {sub_prompt[:50]}...")

            # Databricks-only extraction per workspace policy
            if not os.environ.get('DATABRICKS_API_KEY'):
                logger.error("DATABRICKS_API_KEY not available; cannot perform fact extraction")
                return {
                    "sub_prompt": sub_prompt,
                    "original_analysis": analysis_text,
                    "extracted_facts": [],
                    "extraction_metadata": {
                        "model_used": self.model_id,
                        "total_extractions": 0,
                        "error": "Missing DATABRICKS_API_KEY"
                    }
                }

            # Ensure we use a Databricks model ID
            model_to_use = self.model_id if self.model_id.startswith("databricks") else "databricks-llama-4-maverick"

            try:
                result = lx.extract(
                    text_or_documents=input_text,
                    prompt_description=prompt,
                    examples=examples,
                    model_id=model_to_use,
                )
            except Exception as api_error:
                logger.error(f"Databricks extraction failed: {api_error}")
                return {
                    "sub_prompt": sub_prompt,
                    "original_analysis": analysis_text,
                    "extracted_facts": [],
                    "extraction_metadata": {
                        "model_used": model_to_use,
                        "total_extractions": 0,
                        "error": str(api_error)
                    }
                }

            # Process the results
            if result and hasattr(result, 'extractions') and result.extractions:
                extracted_facts = {
                    "sub_prompt": sub_prompt,
                    "original_analysis": analysis_text,
                    "extracted_facts": [],
                    "extraction_metadata": {
                        "model_used": self.model_id,
                        "total_extractions": len(result.extractions)
                    }
                }

                for extraction in result.extractions:
                    fact = {
                        "category": extraction.extraction_class,
                        "text": extraction.extraction_text,
                        "attributes": extraction.attributes or {}
                    }
                    extracted_facts["extracted_facts"].append(fact)

                logger.info(f"Successfully extracted {len(result.extractions)} facts")
                return extracted_facts
            else:
                logger.warning("No facts extracted from analysis text")
                return None

        except Exception as e:
            logger.error(f"Error during fact extraction: {e}", exc_info=True)
            return None

    def extract_facts_from_multiple_analyses(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract facts from multiple analysis results.

        Args:
            analyses: List of analysis dictionaries containing analysis text and metadata

        Returns:
            List of dictionaries containing extracted facts for each analysis
        """
        results = []

        for analysis in analyses:
            try:
                analysis_text = analysis.get("analysis_summary", "")
                sub_prompt = analysis.get("sub_prompt_analyzed", "")
                context = analysis.get("analysis_context", "")

                if not analysis_text:
                    logger.warning(f"No analysis text found for sub-prompt: {sub_prompt}")
                    continue

                extracted_facts = self.extract_facts_from_analysis(
                    analysis_text=analysis_text,
                    sub_prompt=sub_prompt,
                    context=context
                )

                if extracted_facts:
                    # Add original analysis metadata
                    extracted_facts["original_analysis_metadata"] = {
                        "title": analysis.get("title", ""),
                        "supporting_quotes": analysis.get("supporting_quotes", [])
                    }
                    results.append(extracted_facts)

            except Exception as e:
                logger.error(f"Error processing analysis for fact extraction: {e}")
                continue

        logger.info(f"Completed fact extraction for {len(results)} analyses")
        return results

    def extract_facts_from_text(self, text: str, context: str = "") -> Dict[str, Any]:
        """
        Extract facts from a single text (e.g., analysis section).

        Args:
            text: The text to extract facts from
            context: Additional context about the text

        Returns:
            Dictionary containing extracted facts and metadata
        """
        # Databricks-only: require API key
        if not os.environ.get('DATABRICKS_API_KEY'):
            logger.error("DATABRICKS_API_KEY not available; cannot perform fact extraction")
            return {
                "extracted_facts": [],
                "extraction_metadata": {
                    "model_used": self.model_id,
                    "total_extractions": 0,
                    "error": "Missing DATABRICKS_API_KEY",
                    "context": context
                }
            }

        try:
            # Try a simple text-based approach instead of structured LangExtract
            # This avoids JSON parsing issues with Databricks models
            return self._simple_text_extraction(text, context)

        except Exception as e:
            logger.error(f"Simple text extraction failed: {e}")
            return {
                "extracted_facts": [],
                "extraction_metadata": {
                    "model_used": "simple_text_extraction",
                    "total_extractions": 0,
                    "error": str(e),
                    "context": context
                }
            }

    def _simple_text_extraction(self, text: str, context: str) -> Dict[str, Any]:
        """
        Simple text-based fact extraction using direct Databricks API calls.
        Avoids LangExtract's JSON parsing issues.
        """
        try:
            # Use the Databricks provider directly for a simple completion
            from langextract_databricksprovider import DatabricksProvider

            provider = DatabricksProvider()

            # Create a simple prompt for fact extraction
            prompt = f"""
Extract key facts and their definitions from the following text.
Format each fact as: FACT: [fact name] | DEFINITION: [definition/value]

Text to analyze:
{text}

Facts:"""

            # Make a simple completion call
            response = provider.complete(
                prompt=prompt,
                model_id=self.model_id if self.model_id.startswith("databricks") else "databricks-llama-4-maverick",
                max_tokens=1000,
                temperature=0.1
            )

            # Parse the simple text response
            facts = self._parse_simple_response(response)

            return {
                "extracted_facts": facts,
                "extraction_metadata": {
                    "model_used": self.model_id,
                    "total_extractions": len(facts),
                    "context": context
                }
            }

        except ImportError:
            logger.warning("DatabricksProvider not available, falling back to basic extraction")
            return self._basic_pattern_extraction(text, context)
        except Exception as e:
            logger.error(f"Simple text extraction failed: {e}")
            return self._basic_pattern_extraction(text, context)

    def _parse_simple_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse simple text response into fact items."""
        facts = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            if 'FACT:' in line and 'DEFINITION:' in line:
                try:
                    parts = line.split('|')
                    fact_part = parts[0].replace('FACT:', '').strip()
                    def_part = parts[1].replace('DEFINITION:', '').strip()

                    if fact_part and def_part:
                        facts.append({
                            "text": fact_part,
                            "extraction_text": fact_part,
                            "category": "fact_definition",
                            "extraction_class": "fact_definition",
                            "attributes": {
                                "fact": fact_part,
                                "definition": def_part
                            }
                        })
                except Exception as e:
                    logger.debug(f"Failed to parse line: {line}, error: {e}")
                    continue

        return facts

    def _basic_pattern_extraction(self, text: str, context: str) -> Dict[str, Any]:
        """Fallback pattern-based extraction without LLM calls."""
        import re

        facts = []

        # Pattern 1: "X is Y" or "X means Y"
        patterns = [
            r'([A-Z][A-Za-z\s]+?)\s+(?:is|means)\s+([^.]+)',
            r'([A-Z][A-Za-z\s]+?):\s*([^.]+)',
            r'The\s+([A-Za-z\s]+?)\s+(?:is|shall be)\s+([^.]+)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                fact = match.group(1).strip()
                definition = match.group(2).strip()

                if len(fact) > 2 and len(definition) > 2:
                    facts.append({
                        "text": fact,
                        "extraction_text": fact,
                        "category": "fact_definition",
                        "extraction_class": "fact_definition",
                        "attributes": {
                            "fact": fact,
                            "definition": definition
                        }
                    })

        return {
            "extracted_facts": facts,
            "extraction_metadata": {
                "model_used": "pattern_extraction",
                "total_extractions": len(facts),
                "context": context
            }
        }

    # -------------------- Flexible Fact/Definition Extraction --------------------
    def _extract_fact_definition_pairs(self, item: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Extract fact-definition pairs from LangExtract items using native attributes.
        Leverages LangExtract's natural tagging instead of hard-coded normalization.
        """
        pairs: List[Tuple[str, str]] = []
        text = (item or {}).get("text") or (item or {}).get("extraction_text") or ""
        category = (item or {}).get("category") or (item or {}).get("extraction_class") or ""
        attrs = (item or {}).get("attributes") or {}

        logger.debug(f"Processing item: text='{text}', category='{category}', attrs={attrs}")

        # Method 1: Direct fact-definition attributes (preferred approach)
        fact = attrs.get("fact")
        definition = attrs.get("definition")
        if fact and definition:
            pairs.append((str(fact).strip(), str(definition).strip()))
            logger.debug(f"Found direct fact-definition: {fact} -> {definition}")
            return pairs

        # Method 2: LangExtract identified this as fact_definition class
        if category == "fact_definition":
            # Use the extraction text as fact, look for definition in attributes
            if definition:
                pairs.append((text.strip(), str(definition).strip()))
            elif attrs.get("value"):
                pairs.append((text.strip(), str(attrs["value"]).strip()))
            else:
                # Fallback: use text as both fact and definition if no separate definition
                pairs.append((text.strip(), text.strip()))
            return pairs

        # Method 3: Traditional attribute-based extraction with flexible mapping
        # Look for any attribute that could be a definition
        definition_keys = ["definition", "meaning", "value", "amount", "rate", "percentage", "description"]
        for def_key in definition_keys:
            if def_key in attrs and attrs[def_key]:
                # Use text as fact, attribute as definition
                pairs.append((text.strip(), str(attrs[def_key]).strip()))
                logger.debug(f"Found {def_key} mapping: {text} -> {attrs[def_key]}")
                return pairs

        # Method 4: Use category and text for context-aware extraction
        if category and text.strip():
            # For financial amounts, rates, etc., the text often IS the definition
            if category.lower() in {"financial_amount", "percentage", "rate", "currency", "date"}:
                # Try to infer a meaningful fact name from attributes or context
                fact_name = attrs.get("type") or attrs.get("name") or category
                pairs.append((str(fact_name).strip(), text.strip()))
            else:
                # For other categories, text might be the fact, category the definition
                pairs.append((text.strip(), category.strip()))

            logger.debug(f"Category-based mapping: {text} -> {category}")
            return pairs

        # Method 5: Fallback - if we have meaningful text, create a basic pair
        if text.strip() and len(text.strip()) > 1:
            # Use text as fact, try to find any meaningful attribute as definition
            definition_value = None
            for key, value in attrs.items():
                if value and str(value).strip() and key not in ["extraction_class", "category"]:
                    definition_value = str(value).strip()
                    break

            if definition_value:
                pairs.append((text.strip(), definition_value))
            else:
                # Last resort: use text as both fact and definition
                pairs.append((text.strip(), text.strip()))

            logger.debug(f"Fallback mapping: {text} -> {definition_value or text}")

        return pairs

    def extract_fact_definitions_from_text(self, text: str, context: str = "", *, section_name: str = "", filename: str = "") -> List[Dict[str, str]]:
        """High-level helper: run extraction then normalize to Fact/Definition rows."""
        rows: List[Dict[str, str]] = []

        # Debug logging
        logger.info(f"Starting fact extraction for section: {section_name}")
        logger.debug(f"Text length: {len(text)} characters")

        try:
            res = self.extract_facts_from_text(text=text, context=context) or {}
            logger.info(f"Raw extraction result: {res}")

            items = res.get("extracted_facts", [])
            logger.info(f"Found {len(items)} raw extracted items")

            # Debug: log all raw items
            for i, item in enumerate(items):
                logger.debug(f"Raw item {i}: {item}")

            if not items:
                logger.warning("No items extracted from text")
                return rows

            seen: set[Tuple[str, str]] = set()
            for i, it in enumerate(items):
                logger.debug(f"Processing item {i}: {it}")
                try:
                    pairs = self._extract_fact_definition_pairs(it)
                    logger.debug(f"Extracted {len(pairs)} fact-definition pairs: {pairs}")

                    for fact, definition in pairs:
                        key = (fact.strip(), definition.strip())
                        if key in seen:
                            logger.debug(f"Skipping duplicate: {key}")
                            continue
                        seen.add(key)
                        rows.append({
                            "Filename": filename or "",
                            "Section": section_name or "",
                            "Fact": fact.strip(),
                            "Definition": definition.strip(),
                        })
                        logger.debug(f"Added fact: {fact.strip()} -> {definition.strip()}")
                except Exception as e:
                    logger.error(f"Error extracting fact-definition pairs from item {i}: {e}")
                    continue

            logger.info(f"Final result: {len(rows)} fact definitions extracted")
            return rows

        except Exception as e:
            logger.error(f"Error in extract_fact_definitions_from_text: {e}", exc_info=True)
            return rows

    def extract_fact_definitions_for_results(self, results_with_real_analysis: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> List[Dict[str, str]]:
        """Aggregate Fact/Definition rows across all files/sections from UI results."""
        aggregated: List[Dict[str, str]] = []
        seen_global: set[Tuple[str, str, str]] = set()  # (Filename, Section, Fact)
        for res, ai in results_with_real_analysis:
            filename = res.get("filename", "Unknown File")
            sections = (ai or {}).get("analysis_sections", {}) or {}
            for section_key, section_data in sections.items():
                text = (section_data or {}).get("Analysis", "")
                if not text:
                    continue
                rows = self.extract_fact_definitions_from_text(
                    text=text,
                    context=f"Section: {section_key} in {filename}",
                    section_name=section_key,
                    filename=filename,
                )
                for row in rows:
                    k = (row["Filename"], row["Section"], row["Fact"])
                    if k in seen_global:
                        continue
                    seen_global.add(k)
                    aggregated.append(row)
        return aggregated


