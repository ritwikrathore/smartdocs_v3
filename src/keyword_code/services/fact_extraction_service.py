"""
Fact Extraction Service - Orchestrates the two-step fact extraction process.
Replaces LangExtract with custom LLM-based extraction using Pydantic-AI agents.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from ..config import logger
from ..ai.databricks_llm import get_databricks_llm, DatabricksLLMClient
from ..agents.fact_extraction_agent import (
    FactTypeIdentificationAgent,
    FactExtractionAgent,
    FactType,
    FactTypeAnalysis,
    FactExtractionResult,
    ExtractedFact
)
import re
from io import BytesIO
import pandas as pd


class FactExtractionService:
    """
    Service for extracting structured facts from analysis text using LLM-based agents.
    
    This service orchestrates a two-step process:
    1. Identify expected fact types from the user's query
    2. Extract facts from the analysis text based on identified types
    """
    
    def __init__(self, databricks_client: Optional[DatabricksLLMClient] = None):
        """
        Initialize the fact extraction service.
        
        Args:
            databricks_client: Optional Databricks LLM client. If not provided, will create one.
        """
        self.databricks_client = databricks_client or get_databricks_llm()
        
        if not self.databricks_client:
            raise RuntimeError("Failed to initialize Databricks LLM client for fact extraction")
        
        # Initialize the agents
        self.type_identification_agent = FactTypeIdentificationAgent(self.databricks_client)
        self.extraction_agent = FactExtractionAgent(self.databricks_client)
        
        logger.info("FactExtractionService initialized successfully")
    
    async def extract_facts_async(
        self,
        query: str,
        analysis_text: str,
        context: str = "",
        max_retries: int = 3
    ) -> FactExtractionResult:
        """
        Extract facts from analysis text (async version).
        
        Args:
            query: The original query/sub-prompt
            analysis_text: The analysis text to extract facts from
            context: Additional context about the analysis
            max_retries: Maximum number of retry attempts for extraction
            
        Returns:
            FactExtractionResult with extracted facts
        """
        try:
            # Step 1: Identify expected fact types from the query
            logger.info(f"Identifying fact types for query: {query[:100]}...")
            type_analysis = await self.type_identification_agent.identify_fact_types(
                query=query,
                context=context
            )
            
            logger.info(
                f"Identified fact types: {[ft.value for ft in type_analysis.expected_fact_types]} "
                f"(confidence: {type_analysis.confidence:.2f})"
            )
            
            # Step 2: Extract facts based on identified types
            logger.info(f"Extracting facts from analysis text ({len(analysis_text)} chars)...")
            extraction_result = await self.extraction_agent.extract_facts(
                query=query,
                analysis_text=analysis_text,
                expected_fact_types=type_analysis.expected_fact_types,
                extraction_hints=type_analysis.extraction_hints,
                max_retries=max_retries
            )
            
            # Add type analysis metadata to the result
            extraction_result.extraction_metadata.update({
                "type_analysis_confidence": type_analysis.confidence,
                "type_analysis_reasoning": type_analysis.reasoning,
                "expected_fact_types": [ft.value for ft in type_analysis.expected_fact_types],
                "extraction_hints": type_analysis.extraction_hints
            })
            
            logger.info(
                f"Extracted {len(extraction_result.extracted_facts)} facts "
                f"({len(extraction_result.errors)} errors)"
            )
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error in fact extraction service: {e}", exc_info=True)
            return FactExtractionResult(
                query=query,
                analysis_text=analysis_text,
                extracted_facts=[],
                fact_types_found=[],
                extraction_metadata={
                    "error": str(e)
                },
                errors=[f"Service error: {str(e)}"]
            )
    
    def extract_facts(
        self,
        query: str,
        analysis_text: str,
        context: str = "",
        max_retries: int = 3
    ) -> FactExtractionResult:
        """
        Extract facts from analysis text (synchronous version).
        
        Args:
            query: The original query/sub-prompt
            analysis_text: The analysis text to extract facts from
            context: Additional context about the analysis
            max_retries: Maximum number of retry attempts for extraction
            
        Returns:
            FactExtractionResult with extracted facts
        """
        try:
            # Run the async version in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.extract_facts_async(query, analysis_text, context, max_retries)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous fact extraction: {e}", exc_info=True)
            return FactExtractionResult(
                query=query,
                analysis_text=analysis_text,
                extracted_facts=[],
                fact_types_found=[],
                extraction_metadata={
                    "error": str(e)
                },
                errors=[f"Synchronous wrapper error: {str(e)}"]
            )
    
    def extract_facts_from_text(
        self,
        text: str,
        context: str = "",
        section_name: str = "",
        filename: str = ""
    ) -> Dict[str, Any]:
        """
        Extract facts from a single text section (backward compatible with old interface).
        
        Args:
            text: The text to extract facts from
            context: Additional context about the text
            section_name: Name of the section being processed
            filename: Name of the file being processed
            
        Returns:
            Dictionary containing extracted facts and metadata (compatible with old format)
        """
        try:
            # Use context or section_name as the query hint
            query = context or section_name or "Extract key facts from this text"
            
            # Extract facts
            result = self.extract_facts(
                query=query,
                analysis_text=text,
                context=f"Section: {section_name}, File: {filename}"
            )
            
            # Convert to old format for backward compatibility
            extracted_facts = []
            for fact in result.extracted_facts:
                extracted_facts.append({
                    "text": fact.fact_name,
                    "extraction_text": fact.fact_name,
                    "category": fact.fact_type.value,
                    "extraction_class": fact.fact_type.value,
                    "attributes": {
                        "fact": fact.fact_name,
                        "definition": fact.fact_value,
                        "confidence": fact.confidence,
                        **fact.metadata
                    }
                })
            
            return {
                "extracted_facts": extracted_facts,
                "extraction_metadata": {
                    "model_used": "pydantic-ai-fact-extraction",
                    "total_extractions": len(extracted_facts),
                    "context": context,
                    **result.extraction_metadata
                }
            }
            
        except Exception as e:
            logger.error(f"Error in extract_facts_from_text: {e}", exc_info=True)
            return {
                "extracted_facts": [],
                "extraction_metadata": {
                    "model_used": "pydantic-ai-fact-extraction",
                    "total_extractions": 0,
                    "error": str(e),
                    "context": context
                }
            }
    
    def extract_fact_definitions_from_text(
        self,
        text: str,
        context: str = "",
        section_name: str = "",
        filename: str = ""
    ) -> List[Dict[str, str]]:
        """
        Extract fact-definition pairs from text (backward compatible with old interface).
        
        Args:
            text: The text to extract facts from
            context: Additional context about the text
            section_name: Name of the section being processed
            filename: Name of the file being processed
            
        Returns:
            List of dictionaries with Filename, Section, Fact, and Definition keys
        """
        try:
            # Use context or section_name as the query hint
            query = context or section_name or "Extract key facts and their definitions from this text"
            
            # Extract facts
            result = self.extract_facts(
                query=query,
                analysis_text=text,
                context=f"Section: {section_name}, File: {filename}"
            )
            
            # Convert to fact-definition pairs
            rows = []
            seen = set()
            
            for fact in result.extracted_facts:
                key = (filename, section_name, fact.fact_name)
                if key in seen:
                    continue
                seen.add(key)
                
                rows.append({
                    "Filename": filename or "",
                    "Section": section_name or "",
                    "Fact": fact.fact_name,
                    "Definition": fact.fact_value
                })
            
            logger.info(f"Extracted {len(rows)} fact-definition pairs from {section_name}")
            return rows
            
        except Exception as e:
            logger.error(f"Error in extract_fact_definitions_from_text: {e}", exc_info=True)
            return []
    
    def extract_fact_definitions_for_results(
        self,
        results_with_real_analysis: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> List[Dict[str, str]]:
        """
        Extract fact-definition pairs from multiple analysis results (backward compatible).
        
        Args:
            results_with_real_analysis: List of tuples (result_dict, analysis_dict)
            
        Returns:
            List of dictionaries with Filename, Section, Fact, and Definition keys
        """
        aggregated = []
        seen_global = set()
        
        for res, ai in results_with_real_analysis:
            filename = res.get("filename", "Unknown File")
            sections = (ai or {}).get("analysis_sections", {}) or {}
            
            for section_key, section_data in sections.items():
                text = (section_data or {}).get("Analysis", "")
                if not text:
                    continue
                
                # Extract facts for this section
                rows = self.extract_fact_definitions_from_text(
                    text=text,
                    context=f"Section: {section_key} in {filename}",
                    section_name=section_key,
                    filename=filename
                )
                
                # Add to aggregated results (avoiding duplicates)
                for row in rows:
                    k = (row["Filename"], row["Section"], row["Fact"])
                    if k in seen_global:
                        continue
                    seen_global.add(k)
                    aggregated.append(row)
        
        logger.info(f"Extracted {len(aggregated)} total fact-definition pairs from {len(results_with_real_analysis)} documents")
        return aggregated

    def extract_facts_from_multiple_analyses(
        self,
        analyses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract facts from multiple analysis results (backward compatible with old interface).

        Args:
            analyses: List of analysis dictionaries containing analysis_summary, sub_prompt_analyzed, etc.

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

                # Extract facts
                extraction_result = self.extract_facts(
                    query=sub_prompt,
                    analysis_text=analysis_text,
                    context=context
                )

                # Convert to old format for compatibility
                extracted_facts = {
                    "sub_prompt": sub_prompt,
                    "original_analysis": analysis_text,
                    "extracted_facts": [
                        {
                            "category": fact.fact_type.value,
                            "text": fact.fact_name,
                            "attributes": {
                                "fact": fact.fact_name,
                                "definition": fact.fact_value,
                                **fact.metadata
                            }
                        }
                        for fact in extraction_result.extracted_facts
                    ],
                    "extraction_metadata": extraction_result.extraction_metadata
                }

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


def sanitize_sheet_name(name: str, existing: Optional[set] = None) -> str:
    """
    Sanitize a string to be a valid Excel sheet name.

    - Replaces invalid characters with underscores: \\ / * ? : [ ]
    - Trims to Excel's 31 character limit
    - Handles duplicates by appending " (n)" while keeping within the 31-char limit

    Args:
        name: Original sheet name (usually the document/filename)
        existing: Optional set of already used sheet names to avoid collisions

    Returns:
        A sanitized, unique sheet name safe for Excel
    """
    if existing is None:
        existing = set()

    # Replace forbidden characters
    cleaned = re.sub(r"[\\\\\/*\?:\[\]]", "_", str(name or ""))
    cleaned = cleaned.strip()

    if not cleaned:
        cleaned = "Sheet"

    max_len = 31
    # Base name to use for truncation; leave room for duplicate suffixes
    base = cleaned[:max_len]
    sheet = base
    counter = 1
    while sheet in existing:
        suffix = f" ({counter})"
        allowed = max_len - len(suffix)
        sheet = base[:allowed].rstrip()
        sheet = f"{sheet}{suffix}"
        counter += 1

    # Final safety truncate
    return sheet[:max_len]


def export_fact_definitions_to_excel_bytes(rows: List[Dict[str, Any]]) -> bytes:
    """
    Create a multi-sheet Excel file (bytes) from fact-definition rows.

    Each distinct 'Filename' becomes its own sheet. Sheets contain only two columns:
    'Fact' and 'Definition', preserving the current export format.

    Args:
        rows: List of dicts with keys: Filename, Section, Fact, Definition

    Returns:
        Excel file as bytes
    """
    # Group rows by filename
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        filename = (r.get("Filename") or "Unknown").strip()
        if not filename:
            filename = "Unknown"
        grouped.setdefault(filename, []).append(r)

    buffer = BytesIO()
    used_sheet_names = set()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for filename, group_rows in grouped.items():
            sheet_name = sanitize_sheet_name(filename, used_sheet_names)
            used_sheet_names.add(sheet_name)

            # Build DataFrame with only the two required columns
            df = pd.DataFrame([
                {"Fact": gr.get("Fact", ""), "Definition": gr.get("Definition", "")} for gr in group_rows
            ])

            # Ensure columns are present even if empty
            if df.empty:
                df = pd.DataFrame(columns=["Fact", "Definition"])

            # Write to its own sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    buffer.seek(0)
    return buffer.getvalue()

