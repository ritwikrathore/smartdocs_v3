"""
Configuration settings for the keyword_code package.
"""

import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# Get the project root directory
root_dir = Path(__file__).parent.parent.parent  # This should point to the project root
env_path = root_dir / '.env'

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,  # Temporarily set to DEBUG to help diagnose reranker token issue
    format="%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"  # Added threadName
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file with explicit path
load_dotenv(dotenv_path=env_path)

# Verify if DATABRICKS_API_KEY is loaded
databricks_token = os.environ.get("DATABRICKS_API_KEY")
if databricks_token:
    logger.info("DATABRICKS_API_KEY loaded successfully")
    # Don't log the full token for security reasons
    logger.info(f"Token starts with: {databricks_token[:4]}...")
else:
    logger.error("DATABRICKS_API_KEY not found in environment variables after load_dotenv()")

# --- Worker Configuration ---
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
ENABLE_PARALLEL = os.environ.get("ENABLE_PARALLEL", "true").lower() == "true"

# --- RAG Configuration ---
FUZZY_MATCH_THRESHOLD = 85  # Lowered threshold (0-100) to better handle quotation mark differences
RAG_TOP_K = 15  # Number of relevant chunks to retrieve per sub-prompt
RAG_WORKERS = 4  # Number of workers for parallel RAG processing

# --- Chunking Configuration ---
# Sentence chunker parameters
SENTENCES_PER_CHUNK = 6  # Number of sentences per chunk
MIN_CHUNK_CHAR_LENGTH = 50  # Minimum character length for a chunk to be valid

# --- Model Paths ---
# Using Databricks for all models, no local models needed
# RERANKER_MODEL_PATH is kept for backward compatibility but not used anymore
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH", "src/keyword_code/reranking_model_local")

# --- Interaction Logging Configuration ---
# Set to True to enable detailed logging of BM25, semantic search, reranker, and LLM interactions
ENABLE_INTERACTION_LOGGING = True  # Disabled by default

# --- Databricks Models ---
# Configuration for Databricks services
USE_DATABRICKS_EMBEDDING = True  # Use Databricks for embeddings
USE_DATABRICKS_LLM = True  # Use Databricks for LLM
USE_DATABRICKS_RERANKER = True  # Use Databricks for reranking

# --- Reranker Configuration ---
# The Databricks reranker model has a maximum context window of 512 tokens
# Inputs longer than this will be automatically truncated
RERANKER_MAX_TOKENS = 512  # Maximum token length for the reranker model
RERANKER_API_TIMEOUT = 60  # Timeout in seconds for reranker API calls at startup
ENABLE_LLM_RERANKER_FALLBACK = True  # Enable automatic fallback to LLM-based reranker on API failure

# --- LLM Configuration ---
# Using Databricks LLM
DECOMPOSITION_MODEL_NAME = "databricks-llama-4-maverick"  # Databricks model name
ANALYSIS_MODEL_NAME = "databricks-llama-4-maverick"  # Databricks model name

# --- UI Configuration ---
# Define primary colors
PROCESS_CYAN = "#00ADE4"
DARK_BLUE = "#002345"
LIGHT_BLUE_TINT = "#E6F7FD"  # Example tint (adjust as needed)
VERY_LIGHT_GRAY = "#FAFAFA"

# --- Saved Prompts Configuration ---
# Prompts are organized by mode to keep Ask vs Review suggestions separate.
SAVED_PROMPTS = {
    "Ask": {
        "General Analysis": [
            {
                "label": "Loans Analysis",
                "prompt": (
                    "1. What is the currency of the loan?\n"
                    "2. What is the loan amount for different tranches and loan types such as 'A Loan', 'B1 Loan', 'C Loan'??\n"
                    "3. What is the spread rate or margin rate for different loans?\n"
                    "4. What are the business day definitions?\n"
                    "5. What are the interest payment dates?\n"
                    "6. What are the interest terms, variable or fixed rate? Is it Term SOFR, NON-USD Index, or NCCR for different loans?\n"
                    "7. Interest shall accrue from day to day on what basis?\n"
                    "8. What are the partial prepayment terms / prepayment premium and allocation of principal amounts outstanding.\n"
                    "9. What are the repayment terms and schedule?\n"
                    "10. What are all the fees the borrower shall pay and the amounts?\n"
                    "11. what is the commitment fee on undisbursed amount of the loan?\n"
                    "12. What are the terms for default interest?\n"
                    "13. What is the maturity date?\n"
                    "14. When does the availability period end?"
                ),
            },
            {
                "label": "Equity Analysis",
                "prompt": (
                    "1. What is the name of the issuing company?\n"
                    "2. Who are the investors involved in this transaction?\n"
                    "3. What is the investment commitment amount that IFC (International Finance Corporation) has agreed to in this transaction?\n"
                    "4. What type of equity shares is IFC committing to in this agreement?\n"
                    "5. How many shares or units is IFC subscribing to?\n"
                    "6. What is the price per share or unit for IFC's subscription?\n"
                    "7. What is the signing date of the agreement?\n"
                    "8. Are there any fees or expenses associated with the agreement that affect IFC?\n"
                    "9. What type of expense is it, such as equalization fee, mobilization, advisory, admin fee, etc.?\n"
                    "10. What fees or expenses are explicitly paid to or paid by IFC in this transaction?\n"
                    "11. Does IFC have any special rights or preferences, such as voting rights, dividends, or liquidation preferences, in this agreement?\n"
                    "12. Are there any specific conditions or contingencies related to IFC's participation in the transaction?"
                ),
            },
            {
                "label": "Guarentee Analysis",
                "prompt": (
                    "Check if clauses relating to the following keywords are present in the agreement:\n"
                    "1. Default\n"
                    "2. Restructuring\n"
                    "3. Distressed sale\n"
                    "4. Bankruptcy\n"
                    "5. Rating downgrade"
                ),
            },
        ]
    },
    "Review": {
        "Financial Statement Review": [
            {
                "label": "Numeric Precision Validation",
                "explanation": "All billion-scale values must include decimal precision to avoid ambiguity.",
                "prompt": "Verify that all billion values are expressed with decimal precision (e.g., '1.0 billion' not '1 billion')",
            },
            {
                "label": "Currency Formatting Validation",
                "explanation": "All currency names should use Sentence case (first letter capitalized, rest lowercase).",
                "prompt": "Check that all currency references use proper Sentence case (e.g., 'Indian rupee' not 'Indian Rupee' or 'indian rupee')",
            },
            {
                "label": "Language Intent and Spelling Validation",
                "explanation": "Detect commonly confused words that change meaning due to spelling errors.",
                "prompt": "Identify potential word confusion errors such as 'decease' vs 'decrease', 'principal' vs 'principle', 'affect' vs 'effect'",
            },
            {
                "label": "Percentage Rate Formatting",
                "explanation": "Interest rates and percentages should include a percent symbol and at least one decimal place for precision.",
                "prompt": "Confirm that percentage rates include a '%' symbol and at least one decimal place (e.g., '5.5%' not '5%')",
            },
            {
                "label": "Thousands Separator for Large Numbers",
                "explanation": "Numbers greater than or equal to 1,000 should include thousands separators for readability.",
                "prompt": "Flag numbers >= 1,000 that lack thousands separators (e.g., '10000' should be '10,000')",
            },
            {
                "label": "ISO Currency Codes Formatting",
                "explanation": "When currency codes are used, require uppercase three-letter ISO codes without punctuation.",
                "prompt": "Ensure ISO currency codes are uppercase three-letter codes when used as codes (e.g., 'USD 1,000,000' not 'Usd 1,000,000' or '$US 1,000,000')",
            },
            {
                "label": "Date Format Consistency",
                "explanation": "Dates should follow one consistent format across the document.",
                "prompt": "Ensure all dates follow the 'Month DD, YYYY' format (e.g., 'June 30, 2024' not '30/06/2024')",
            },
            {
                "label": "Calculation Verification (Totals)",
                "explanation": "Totals should equal the sum of their components, allowing a small rounding tolerance.",
                "prompt": "Verify that 'Total Liabilities' equals 'Current Liabilities' + 'Non-current Liabilities' within a rounding tolerance of 1 unit",
            },
            {
                "label": "Units Consistency (Thousands/Millions)",
                "explanation": "Amounts within a section should use a consistent presentation unit (e.g., thousands or millions) and the unit must be stated.",
                "prompt": "Check that amounts are consistently reported in the stated presentation unit (e.g., 'US$ in millions'); flag mixed units or missing unit declarations",
            },
            {
                "label": "Negative Numbers Formatting",
                "explanation": "Negative amounts should be displayed in parentheses rather than using a leading minus sign.",
                "prompt": "Ensure negative amounts are displayed in parentheses (e.g., '(1,234)' not '-1,234')",
            },
            {
                "label": "Comprehensive Financial Statement Validation",
                "explanation": "Comprehensive checklist covering numeric formatting, currency conventions, terminology accuracy, calculation verification, and consistency checks.",
                "prompt": """1) Verify that all billion values are expressed with decimal precision (e.g., '1.0 billion' not '1 billion')
2) Check that all currency references use proper Sentence case (e.g., 'Indian rupee' not 'Indian Rupee' or 'indian rupee')
3) Identify potential word confusion errors such as 'decease' vs 'decrease', 'principal' vs 'principle', 'affect' vs 'effect'
4) Confirm that percentage rates include a '%' symbol and at least one decimal place (e.g., '5.5%' not '5%')
5) Flag numbers >= 1,000 that lack thousands separators (e.g., '10000' should be '10,000')
6) Ensure ISO currency codes are uppercase three-letter codes when used as codes (e.g., 'USD 1,000,000' not 'Usd 1,000,000' or '$US 1,000,000')
7) Ensure all dates follow the 'Month DD, YYYY' format (e.g., 'June 30, 2024' not '30/06/2024')
8) Verify that 'Total Liabilities' equals 'Current Liabilities' + 'Non-current Liabilities' within a rounding tolerance of 1 unit
9) Check that amounts are consistently reported in the stated presentation unit (e.g., 'US$ in millions'); flag mixed units or missing unit declarations
10) Ensure negative amounts are displayed in parentheses (e.g., '(1,234)' not '-1,234')""",

            },

        ]
    },
}
