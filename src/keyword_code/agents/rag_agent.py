"""
Pydantic-AI agent for intelligent RAG optimization and retry functionality.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
import os

from ..config import logger
from ..rag.retrieval import retrieve_relevant_chunks_async
from ..ai.databricks_llm import DatabricksLLMClient
from ..utils.interaction_logger import log_rag_parameters


class RAGContext(BaseModel):
    """Context for RAG operations."""
    query: str
    chunks: List[Dict[str, Any]]
    embedding_model: Any
    reranker_model: Any = None
    precomputed_embeddings: Any = None
    valid_chunk_indices: List[int] = Field(default_factory=list)
    current_results: List[Dict[str, Any]] = Field(default_factory=list)
    bm25_weight: float = 0.5
    semantic_weight: float = 0.5
    top_k: int = 10
    bm25_terms: List[str] = Field(default_factory=list)
    hyde_phrases: List[str] = Field(default_factory=list)


class RAGAnalysis(BaseModel):
    """Analysis of current RAG results."""
    query_type: str = Field(description="Type of query (e.g., 'MT599_Swift', 'general_legal', 'financial_terms')")
    current_quality_score: float = Field(description="Quality score of current results (0-1)")
    issues_identified: List[str] = Field(description="List of issues with current results")
    recommended_bm25_weight: float = Field(description="Recommended BM25 weight (0-1)")
    recommended_semantic_weight: float = Field(description="Recommended semantic weight (0-1)")
    recommended_top_k: int = Field(description="Recommended number of results to retrieve")
    recommended_bm25_terms: List[str] = Field(default_factory=list, description="Lexical terms to supply to BM25 in the retry")
    recommended_hyde_phrases: List[str] = Field(default_factory=list, description="Speculative HYDE prompts to seed semantic retrieval")
    reasoning: str = Field(description="Explanation of the recommendations")


class RAGRetryParams(BaseModel):
    """Parameters for RAG retry."""
    bm25_weight: float = Field(description="Weight for BM25 search (0-1)")
    semantic_weight: float = Field(description="Weight for semantic search (0-1)")
    top_k: int = Field(description="Number of top results to retrieve")
    reasoning: str = Field(description="Reasoning for these parameters")


class RAGOptimizationAgent:
    """
    Pydantic-AI agent for optimizing RAG retrieval parameters based on query analysis.
    """
    
    def __init__(self, databricks_client: DatabricksLLMClient):
        self.databricks_client = databricks_client
        
        # Create a custom model wrapper for Databricks
        self.model = self._create_databricks_model()
        
        # Create the agent
        self.agent = Agent(
            model=self.model,
            result_type=RAGAnalysis,
            system_prompt="""You are an expert RAG (Retrieval-Augmented Generation) optimization agent.

    Your job is to analyze queries and current RAG results to recommend optimal retrieval parameters.

    Key guidelines:
    1. For keyword-dense queries: Prefer BM25 (weight ~0.7-0.8) and supply explicit lexical terms the search system should OR together.
    2. For general legal queries: Balanced approach (BM25 ~0.5, semantic ~0.5).
    3. For financial terms: Slightly favor BM25 (weight ~0.6) for precise terminology.
    4. For conceptual queries: Favor semantic search (weight ~0.6-0.7) and provide HYDE-style speculative sentences to seed semantic retrieval.

    Always return:
    - Recommended BM25 / semantic weights that sum to ~1.0.
    - A top_k value aligned with query breadth (default 10 unless you have a strong reason).
    - `recommended_bm25_terms`: ordered list of literal phrases to hand to BM25 (use [] if no change is needed).
    - `recommended_hyde_phrases`: at most 2 concise sentences that sound like excerpts from the document and should guide semantic recall (use [] when not helpful).
    - Clear reasoning that justifies the adjustments referencing current retrieval quality.

    Be concise but thorough in your analysis."""
        )
    
    def _create_databricks_model(self):
        """Create an OpenAIModel configured for Databricks Serving (OpenAI-compatible)."""
        try:
            # Lazy import to avoid circulars
            from ..ai.databricks_llm import DATABRICKS_BASE_URL, DATABRICKS_LLM_MODEL

            api_key = os.environ.get("DATABRICKS_API_KEY")
            if not api_key:
                raise RuntimeError("DATABRICKS_API_KEY is not set in environment variables")

            # OpenAIModel accepts base_url and api_key directly
            return OpenAIModel(
                model_name=DATABRICKS_LLM_MODEL,
                base_url=DATABRICKS_BASE_URL,
                api_key=api_key
            )
        except Exception as e:
            # Log the error but don't fail startup
            logger.error(f"Error using Databricks LLM: {e}")
            return None

    async def analyze_and_recommend(
        self, 
        context: RAGContext
    ) -> RAGAnalysis:
        """
        Analyze the current RAG context and recommend optimization parameters.
        """
        try:
            # Prepare analysis prompt
            current_results_summary = self._summarize_results(context.current_results)
            current_terms = ", ".join(context.bm25_terms) if context.bm25_terms else "<none>"
            current_hyde = "; ".join(context.hyde_phrases) if context.hyde_phrases else "<none>"
            
            prompt = f"""
            Analyze this RAG retrieval scenario:
            
            Query: "{context.query}"
            Current BM25 weight: {context.bm25_weight}
            Current semantic weight: {context.semantic_weight}
            Current top_k: {context.top_k}
            Current BM25 terms: {current_terms}
            Current HYDE phrases: {current_hyde}
            
            Current results summary:
            {current_results_summary}
            
            Please analyze the query type and current results quality, then recommend optimal parameters.
            """
            
            # Run the agent
            result = await self.agent.run(prompt, message_history=[])
            return result.output

        except Exception as e:
            logger.error(f"Error in RAG analysis: {e}", exc_info=True)
            # Return default analysis
            return RAGAnalysis(
                query_type="general",
                current_quality_score=0.5,
                issues_identified=["Analysis failed"],
                recommended_bm25_weight=0.5,
                recommended_semantic_weight=0.5,
                recommended_top_k=context.top_k,
                recommended_bm25_terms=context.bm25_terms,
                recommended_hyde_phrases=context.hyde_phrases,
                reasoning="Default parameters due to analysis error"
            )
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Summarize current RAG results for analysis."""
        if not results:
            return "No results retrieved"
        
        summary_parts = [
            f"Retrieved {len(results)} results",
            f"Average score: {sum(r.get('score', 0) for r in results) / len(results):.3f}",
            f"Score range: {min(r.get('score', 0) for r in results):.3f} - {max(r.get('score', 0) for r in results):.3f}"
        ]
        
        # Add sample of result texts (first 100 chars of each)
        sample_texts = [r.get('text', '')[:100] + '...' for r in results[:3]]
        if sample_texts:
            summary_parts.append(f"Sample results: {'; '.join(sample_texts)}")
        
        return "; ".join(summary_parts)


class RAGRetryTool:
    """
    Tool for retrying RAG with optimized parameters.
    """
    
    def __init__(self, optimization_agent: RAGOptimizationAgent):
        self.optimization_agent = optimization_agent
    
    async def retry_with_optimization(
        self,
        context: RAGContext
    ) -> Tuple[List[Dict[str, Any]], RAGAnalysis]:
        """
        Retry RAG retrieval with optimized parameters.
        
        Returns:
            Tuple of (new_results, analysis)
        """
        try:
            # Get optimization recommendations
            analysis = await self.optimization_agent.analyze_and_recommend(context)

            logger.info(f"RAG optimization analysis: {analysis.reasoning}")
            logger.info(f"Recommended weights - BM25: {analysis.recommended_bm25_weight:.2f}, Semantic: {analysis.recommended_semantic_weight:.2f}")
            logger.info(f"Query type identified: {analysis.query_type}")
            if analysis.recommended_bm25_terms:
                logger.info(f"Recommended BM25 terms: {analysis.recommended_bm25_terms}")
            if analysis.recommended_hyde_phrases:
                logger.info(f"Recommended HYDE phrases: {analysis.recommended_hyde_phrases}")

            # Log RAG parameters from retry agent
            log_rag_parameters(
                sub_prompt_title=f"Retry: {context.query[:50]}...",
                sub_prompt=context.query,
                bm25_weight=analysis.recommended_bm25_weight,
                semantic_weight=analysis.recommended_semantic_weight,
                bm25_terms=analysis.recommended_bm25_terms or context.bm25_terms,
                reasoning=analysis.reasoning,
                source="retry_agent"
            )

            # Apply recommended parameters
            optimized_context = context.model_copy()
            optimized_context.bm25_weight = analysis.recommended_bm25_weight
            optimized_context.semantic_weight = analysis.recommended_semantic_weight
            optimized_context.top_k = analysis.recommended_top_k
            if analysis.recommended_bm25_terms:
                optimized_context.bm25_terms = analysis.recommended_bm25_terms
            if analysis.recommended_hyde_phrases:
                optimized_context.hyde_phrases = analysis.recommended_hyde_phrases

            # Retry RAG with optimized parameters
            new_results = await self._retrieve_with_weights(optimized_context)

            logger.info(f"RAG retry completed: {len(new_results)} results with optimized parameters (BM25: {analysis.recommended_bm25_weight:.2f}, Semantic: {analysis.recommended_semantic_weight:.2f})")

            return new_results, analysis
            
        except Exception as e:
            logger.error(f"Error in RAG retry: {e}", exc_info=True)
            return context.current_results, RAGAnalysis(
                query_type="error",
                current_quality_score=0.0,
                issues_identified=[f"Retry failed: {str(e)}"],
                recommended_bm25_weight=context.bm25_weight,
                recommended_semantic_weight=context.semantic_weight,
                recommended_top_k=context.top_k,
                recommended_bm25_terms=context.bm25_terms,
                recommended_hyde_phrases=context.hyde_phrases,
                reasoning="Retry failed, keeping original parameters"
            )
    
    async def _retrieve_with_weights(self, context: RAGContext) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with weighted hybrid search.
        """
        results = await retrieve_relevant_chunks_async(
            prompt=context.query,
            chunks=context.chunks,
            model=context.embedding_model,
            top_k=context.top_k,
            precomputed_embeddings=context.precomputed_embeddings,
            valid_chunk_indices=context.valid_chunk_indices,
            reranker_model=context.reranker_model,
            bm25_weight=context.bm25_weight,
            semantic_weight=context.semantic_weight,
            bm25_terms=context.bm25_terms,
            alternate_hyde_queries=context.hyde_phrases,
        )

        return results