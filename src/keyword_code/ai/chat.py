"""
Chat functionality.
"""

from typing import Any, Dict, List
from ..config import logger, ANALYSIS_MODEL_NAME, USE_DATABRICKS_LLM


async def generate_chat_response(
    analyzer, user_prompt: str, relevant_chunks: List[Dict[str, Any]]
) -> str:
    """
    Generates a conversational response to a user query based on relevant chunks
    retrieved from multiple documents using RAG.

    Args:
        analyzer: The DocumentAnalyzer instance to use for LLM calls.
        user_prompt: The user's chat message.
        relevant_chunks: A list of dictionaries, each containing chunk details
                         ('filename', 'text', 'page_num', 'score').

    Returns:
        A string containing the AI's conversational response, potentially including
        citations like (Source: filename.pdf, Page: 5).
    """
    if not relevant_chunks:
        logger.warning(f"No relevant context found for chat prompt: '{user_prompt[:50]}...'. Returning default response.")
        return "I couldn't find specific information related to your question in the provided documents. Could you try rephrasing or asking something else?"

    # --- Format Context for LLM ---
    context_str = "\n\n---\n\n".join(
        f"Source: {chunk.get('filename', 'Unknown')}, Page: {chunk.get('page_num', -1) + 1}\n"
        f"Score: {chunk.get('score', 0):.3f}\n"
        f"Content: {chunk.get('text', '')}"
        for chunk in relevant_chunks
    )

    # --- Define System Prompt for Chat ---
    system_prompt = f"""You are a helpful AI assistant designed to answer questions about documents. You will be given a user's question and relevant excerpts from one or more documents.

Your task is:
1. Understand the user's question.
2. Analyze the provided document excerpts to find the answer.
3. Generate a clear, concise, and conversational response based *only* on the information in the excerpts.
4. **Crucially, you MUST cite your sources.** When you use information from an excerpt, add an inline citation immediately after the information, formatted *exactly* as: `(Source: [filename], Page: [page_number])`. Use the filename and page number provided with each excerpt.
5. If multiple excerpts support a statement, you can list multiple citations like `(Source: doc1.pdf, Page: 2)(Source: doc2.pdf, Page: 5)`.
6. If the provided excerpts do not contain the answer to the user's question, explicitly state that you couldn't find the information in the provided context.
7. Do not make assumptions or provide information not present in the excerpts.
8. Respond directly to the user's question without preamble like "Based on the context...".

Example Excerpt Format:
Source: contract_A.pdf, Page: 5
Score: 0.850
Content: The termination clause allows for a 30-day notice period.

Example Response:
The contract allows for a 30-day notice period (Source: contract_A.pdf, Page: 5)."""

    # --- Construct Messages for LLM ---
    human_prompt = f"""User Question: {user_prompt}

Relevant Document Excerpts:
---
{context_str}
---

Please answer the user's question based *only* on these excerpts and cite your sources accurately using the format (Source: [filename], Page: [page_number])."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]

    try:
        logger.info(f"Sending chat request for prompt '{user_prompt[:50]}...' to AI ({ANALYSIS_MODEL_NAME}). Context length: {len(context_str)} chars.")
        response_content = await analyzer._get_completion(messages, model_name=ANALYSIS_MODEL_NAME)
        logger.info(f"Received chat response for prompt '{user_prompt[:50]}...'.")
        # Basic cleaning (optional)
        response_content = response_content.strip()
        return response_content

    except TimeoutError:
        logger.error(f"Chat request timed out for prompt '{user_prompt[:50]}...'.")
        return "I apologize, but the request timed out while generating a response. Please try again."
    except Exception as e:
        logger.error(f"Error during chat AI call for prompt '{user_prompt[:50]}...': {str(e)}", exc_info=True)
        return f"Sorry, I encountered an error while trying to generate a response: {str(e)}"
