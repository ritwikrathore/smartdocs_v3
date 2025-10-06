# CNT SmartDocs Page

A Streamlit page providing AI-powered document intelligence capabilities, designed to be integrated into a multi-page Streamlit application.

## Project Structure

```
.
├── src/
│   └── keyword_code/               # Core document analysis backend
│       ├── ai/                     # AI integration modules
│       │   ├── analyzer.py         # Document analysis with LLM
│       │   ├── chat.py             # Chat functionality
│       │   ├── databricks_llm.py   # Databricks LLM client
│       │   ├── decomposition.py    # Prompt decomposition
│       ├── assets/                 # Application assets (images, logos)
│       ├── models/                 # Model integration
│       │   ├── databricks_embedding.py  # Databricks embedding API client
│       │   ├── databricks_reranker.py  # Databricks reranker API client
│       │   └── embedding.py        # Embedding model interface
│       ├── processors/             # Document processors
│       │   ├── pdf_processor.py    # PDF parsing, Verification, Highlighting.
│       │   └── word_processor.py   # DOCX parsing and conversion to PDF
│       ├── rag/                    # Retrieval-Augmented Generation
│       │   ├── retrieval.py        # Hybrid retrieval system (BM25 + semantic)
│       │   └── chunking.py         # Isolates chunking process
│       ├── utils/                  # Utility functions
│       │   ├── async_utils.py      # Asynchronous processing utilities
│       │   ├── display.py          # UI display functions
│       │   ├── file_manager.py     # Temporary file management
│       │   ├── helpers.py          # General helper functions
│       │   ├── interaction_logger.py # Logging for search and LLM interactions
│       │   ├── memory_monitor.py   # Memory usage monitoring
│       │   ├── spacy_utils.py      # spaCy model management
│       │   └── ui_helpers.py       # Streamlit UI helper functions
│       ├── app.py                  # Main application logic
│       └── config.py               # Configuration settings
├── models/                         # Local storage for downloaded models
│   └── spacy/                      # Local spaCy models
├── tmp/                            # Temporary file storage
├── logs/                           # Log files directory (temporary, to be disabled before moving to prod)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Core Components

*   **`pages/1_📄_CNT_space.py`**: This is the main user interface built with Streamlit. It handles:
    *   Uploading multiple documents (PDF, DOCX).
    *   Accepting user prompts for analysis.
    *   Displaying analysis results, including verified phrases and annotations.
    *   Providing an interactive chat interface for querying documents.
    *   Showing an interactive PDF viewer.
    *   Orchestrating the backend logic by calling functions from the keyword_code modules.

*   **`src/keyword_code/app.py`**: This script contains the main application logic and orchestrates the various components. Its responsibilities include:
    *   Coordinating document preprocessing and analysis.
    *   Managing the Streamlit UI state and interactions.
    *   Handling parallel processing of multiple documents.
    *   Memory management to prevent application slowdowns.
    *   Coordinating the RAG workflow across multiple modules.

*   **`src/keyword_code/ai/`**: This directory contains modules for AI integration:
    *   **`analyzer.py`**: Implements the DocumentAnalyzer class for LLM-based document analysis.
    *   **`databricks_llm.py`**: Client for interacting with Databricks LLM API.
    *   **`decomposition.py`**: Handles breaking down complex prompts into sub-prompts.
    *   **`chat.py`**: Manages chat functionality with document context.

*   **`src/keyword_code/models/`**: This directory contains modules for AI models:
    *   **`databricks_embedding.py`**: Client for Databricks embedding API.
    *   **`databricks_reranker.py`**: Client for Databricks reranker API with token-based truncation.
    *   **`embedding.py`**: Interface for loading and using embedding models.

*   **`src/keyword_code/rag/`**: This directory contains the Retrieval-Augmented Generation implementation:
    *   **`retrieval.py`**: Implements the hybrid retrieval system that combines:
        *   BM25 for fast keyword-based retrieval of relevant document chunks.
        *   Semantic search using embeddings for context-based retrieval.
        *   Databricks reranker for improved relevance scoring with automatic token limit handling.
    *   **`chunking.py`**: Isolates the chunking process.


*   **`src/keyword_code/processors/`**: This directory contains document processing modules:
    *   **`pdf_processor.py`**: Handles PDF parsing, chunking, and annotation.
    *   **`word_processor.py`**: Handles DOCX parsing and conversion.

*   **`src/keyword_code/utils/`**: This directory contains utility modules for:
    *   UI helpers and styling
    *   File management
    *   Memory monitoring
    *   Display functions
    *   Asynchronous utilities

## LLM Integration Points

All interactions with the Large Language Model (LLM) are handled within the `DocumentAnalyzer` class located in `src/keyword_code/ai/analyzer.py`.

The key methods utilizing the LLM are:

1.  **`decompose_prompt(analyzer, user_prompt)`**: Sends the user's initial prompt to the LLM to break it down into smaller, distinct sub-prompts or tasks.

2.  **`analyze_document_with_all_contexts(filename, main_prompt, sub_prompts_with_contexts)`**: Sends all sub-prompts and their relevant contexts to the LLM in a single call. This unified approach allows the LLM to see the whole picture and provide more coherent answers by cross-referencing information between sub-prompts.

3.  **`generate_chat_response(analyzer, user_prompt, relevant_chunks)`**: Sends the user's chat query and relevant context (from RAG across all documents) to the LLM to generate a conversational response with source citations.

**Current LLM Integration:**

The application uses the **Databricks API** to access the LLama 3 model. The implementation is in `src/keyword_code/ai/databricks_llm.py`.

Key components:

*   **`DatabricksLLMClient`**: A client class that wraps the OpenAI SDK configured to use Databricks endpoints.
*   **`get_databricks_llm()`**: A cached function that creates and initializes the client.
*   **`get_completion(messages, max_tokens)`**: Method that formats messages and calls the Databricks API.
*   **`get_completion_async(messages, max_tokens)`**: Async wrapper for the above method.

The core logic for making the actual API call resides in the private method `_get_completion(messages, model_name)` of the `DocumentAnalyzer` class, which calls the Databricks client.

**To use a different LLM API:**

*   Create a new client class similar to `DatabricksLLMClient` in a separate module.
*   Ensure it has a `get_completion_async` method that takes messages and returns text.
*   Update the `DocumentAnalyzer` class to use your new client.
*   Update the configuration in `config.py` to point to your model.

## Reranker Integration and Token Management

The application uses the **Databricks Reranker API** for improving the relevance of retrieved document chunks. The implementation includes sophisticated token management to handle the model's 512 token limit.

**Key Components:**

*   **`DatabricksRerankerModel`**: A wrapper class that provides a CrossEncoder-compatible interface for the Databricks reranker API.
*   **Token-Based Truncation**: Uses a BERT tokenizer to accurately count tokens and ensure inputs don't exceed the 512 token limit.
*   **Proportional Allocation**: Intelligently allocates tokens between query (10-30%) and document (70-90%) text to preserve maximum context.
*   **Graceful Degradation**: Falls back to character-based truncation if the tokenizer is unavailable.

**Configuration:**

The reranker behavior is controlled by the `RERANKER_MAX_TOKENS` setting in `config.py` (default: 512). The system automatically:

*   Counts tokens using a BERT tokenizer for accuracy
*   Truncates at sentence or word boundaries when possible
*   Logs truncation operations for monitoring
*   Maintains semantic coherence of truncated text

**Testing: (Not part of PROD)**

A dedicated test script is available in `databricks_reranker_test/databricks_reranker.py` that includes the same token management functionality for testing the reranker independently.


## Running the Page

This Streamlit page (`1_📄_CNT_space.py`) is designed to be part of a larger multi-page Streamlit application. Place the `pages` directory (containing this file) and the `src` directory within the root directory of the target Streamlit application.

The host application's main entry point (e.g., a `Home.py` or similar) will typically be run using:
```bash
streamlit run <your_main_app_entry_point>.py
```
The "📄 CNT space" page should then appear in the Streamlit sidebar navigation.

## Features

*   **Multi-Document Upload**: Process multiple PDF and DOCX files simultaneously.
*   **AI-Powered Analysis**: Get detailed analysis based on your specific prompts using advanced AI models.
*   **Unified RAG Approach**: Sends all sub-prompts and their contexts to the LLM in a single call, allowing for cross-referencing and more coherent answers.
*   **Hybrid RAG Retrieval**: Combines BM25 keyword search, semantic search, and Databricks reranker for highly relevant context selection.
*   **Databricks Integration**: Uses Databricks API for LLM (LLama 3), embeddings, and reranking, providing high-quality results with automatic token limit handling.
*   **Prompt Decomposition**: Automatically breaks down complex prompts into smaller, manageable analysis tasks.
*   **Phrase Verification**: Automatically verifies AI-extracted supporting phrases against the source documents using fuzzy matching.
*   **PDF Annotation**: Highlights verified phrases directly within the PDF viewer.
*   **Interactive Chat**: Engage in a conversation with your uploaded documents, with responses citing specific sources (filename and page number).
*   **Memory Management**: Implements automatic memory cleanup to prevent application slowdowns and crashes.
*   **Export Results**: Download the analysis findings and supporting citations in Excel (`.xlsx`) or Word (`.docx`) formats.
*   **Export Annotated PDFs**: Download PDFs with verified phrases highlighted.

## Recent Updates

*   **Reranker Token Limit Fix**: Implemented robust token-based truncation for the Databricks reranker model to handle the 512 token limit:
    *   Added BERT tokenizer for accurate token counting instead of character-based approximations
    *   Implemented proportional token allocation between query (10-30%) and document (70-90%) text
    *   Added automatic truncation with intelligent boundary detection (sentence/word boundaries)
    *   Configured `RERANKER_MAX_TOKENS = 512` in config for centralized token limit management
    *   Added comprehensive logging for truncation operations and token usage
*   **Improved Code Organization**: Moved LLM reranking functionality from `rag/retrieval.py` to `ai/llm_reranker.py` for better code organization, keeping AI-related functionality in the appropriate directory.
*   **Streamlined Document Analysis**: Removed legacy `analyze_document` method in favor of the unified `analyze_document_with_all_contexts` approach, which provides more coherent analysis by allowing cross-referencing between sub-prompts.
*   **Enhanced PDF Processing**: Implemented a sentence-based chunking strategy using spaCy in the PDF processor, creating more semantically meaningful chunks (3 sentences per chunk) for improved retrieval.
*   **Consistent Word Document Handling**: Word documents are now converted to PDF format and processed using the same chunking strategy as PDF files, ensuring consistent handling across document types.

## Memory and File Management

The application includes robust memory and file management systems to ensure it runs efficiently without affecting the larger application it's integrated into:

*   **Isolated Temporary Storage**: All temporary files are stored in the app's own `tmp/` directory, which is created if it doesn't exist.
*   **Automatic Cleanup**: The application automatically cleans up temporary files after they're no longer needed, and periodically removes old files.
*   **Memory Monitoring**: A memory monitor tracks usage and automatically frees large objects from the Streamlit session state when memory usage exceeds configurable thresholds to prevent app crashes.
*   **Contained SpaCy Cleanup**: The SpaCy model cleanup only affects the standard SpaCy cache directory, not the models stored in the app's `models/spacy/` directory.
*   **Session State Management**: Large objects in the Streamlit session state are selectively cleared when memory pressure is high, with priority given to objects that can be regenerated.

These systems ensure that the application:
1. Doesn't leak temporary files
2. Manages its memory footprint efficiently
3. Doesn't interfere with the host application's resources
4. Can run for extended periods without degradation

### Chunking Test Framework (Not part of PROD)

The `chunking_test` directory contains a framework for testing different document chunking strategies specifically designed for legal documents. This is particularly useful for improving retrieval of short, important phrases like definitions.

Features:
- Tests 6 different chunking strategies (fixed-size, sentence-based, recursive, semantic, hybrid, rolling window)
- Evaluates how well each strategy preserves important legal definitions
- Includes a hybrid approach that applies different chunking rules to different document sections

To use:
1. Place your legal document in `chunking_test/test_file/` or run streamlit app
2. Run `cd chunking_test && python chunking_test.py`
3. Check results in the `chunking_test/results/` directory

The hybrid chunking approach is particularly effective for legal documents as it uses:
- Single-sentence chunks for definition sections
- 2-3 sentence chunks for articles and operational text
- Larger chunks for narrative or background sections

See `chunking_test/README.md` for more details.
