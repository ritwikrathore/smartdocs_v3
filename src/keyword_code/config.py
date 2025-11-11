"""
Configuration settings for the keyword_code package.
"""

import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# Get the project root directory
root_dir = Path(__file__).parent.parent.parent  # This should point to the project root
env_path = root_dir / '.env'

# --- Logging Configuration ---
# Set to True to enable application log files (app_*.log)
ENABLE_APP_LOGGING = False  # Disabled by default - only console logging when False

# Create logs directory if it doesn't exist
logs_dir = root_dir / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure handlers based on ENABLE_APP_LOGGING
handlers = [logging.StreamHandler()]  # Always have console output
if ENABLE_APP_LOGGING:
    # Create a timestamped log file for all application logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    app_log_file = logs_dir / f"app_{timestamp}.log"
    handlers.append(logging.FileHandler(app_log_file, mode='a', encoding='utf-8'))

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers
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
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 3))
ENABLE_PARALLEL = os.environ.get("ENABLE_PARALLEL", "true").lower() == "true"

# --- RAG Configuration ---
FUZZY_MATCH_THRESHOLD = 85  # Lowered threshold (0-100) to better handle quotation mark differences
RAG_TOP_K = 15  # Number of relevant chunks to retrieve per sub-prompt
RAG_WORKERS = 3  # Number of workers for parallel RAG processing

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
ENABLE_INTERACTION_LOGGING = False  # Disabled by default

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

# --- LLM Retry Configuration ---
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", 3))

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
                    "1. What is the Investment Number for this Project? \n"
                    "2. When is this document Dated? \n"
                    "3. What is the currency of the loan? \n"
                    "4. What is the loan amount for different tranches and loan types such as 'A Loan', 'B1 Loan', 'C Loan'? \n"
                    "5. What is the spread rate or margin rate for different loans? \n"
                    "6. What are the business day definitions? \n"
                    "7. What is the applicable business day convention for adjusting the interest payment date if the scheduled date falls on a non-business day? \n"
                    "8. What are the interest payment dates? \n"
                    "9. What are the interest terms, variable or fixed rate? Is it Term SOFR, NON-USD Index, or NCCR for different loans? \n"
                    "10. What rounding adjustments, if any, are required for the interest rates? \n"
                    "11. Interest shall accrue from day to day on what basis? \n"
                    "12. What are the terms for partial prepayment / prepayment premium and allocation of principal amounts outstanding? \n"
                    "13. What is the method for applying prepayments—is it pro rata basis or in inverse order of maturity? \n"
                    "14. What are the repayment terms and can you list the full repayment schedule with dates? \n"
                    "15. What are all the fees the borrower shall pay and the percentages / amounts? \n"
                    "16. what is the commitment fee percentage on undisbursed amount of the loan? \n"
                    "17. What are the terms for default interest? \n"
                    "18. What is the maturity date? \n"
                    "19. When does the availability period end? \n"
                ),
            },
            {
                "label": "Equity Analysis",
                "prompt": (
                    "1. What is the name of the issuing company? \n"
                    "2. Who are the investors involved in this transaction? \n"
                    "3. What is the investment commitment amount that IFC (International Finance Corporation) has agreed to in this transaction? \n"
                    "4. What type of equity shares is IFC committing to in this agreement? \n"
                    "5. How many shares or units is IFC subscribing to? \n"
                    "6. What is the price per share or unit for IFC's subscription? \n"
                    "7. What is the signing date of the agreement? \n"
                    "8. Are there any fees or expenses associated with the agreement that affect IFC? \n"
                    "9. What type of expense is it, such as equalization fee, mobilization, advisory, admin fee, etc.? \n"
                    "10. What fees or expenses are explicitly paid to or paid by IFC in this transaction? \n"
                    "11. Does IFC have any special rights or preferences, such as voting rights, dividends, or liquidation preferences, in this agreement? \n"
                    "12. Are there any specific conditions or contingencies related to IFC's participation in the transaction? \n"
                ),
            },
            {
                "label": "Guarentee Analysis",
                "explanation": "CNTPI Guarentee Agreement Analysis",
                "prompt": (
                    "Check if clauses relating to the following keywords are present in the agreement: \n"
                    "1. Default \n"
                    "2. Restructuring \n"
                    "3. Distressed sale \n"
                    "4. Bankruptcy \n"
                    "5. Rating downgrade \n"
                ),
            },
            {
                "label": "ASC 320 Analysis",
                "explanation": "ASC 320 Debt Security analysis whose legal form is debt i.e., Bonds, Subscription agreements, Debenture Trust Deed, Note Purchase Agreement, Note Pricing Supplement, etc.",
                "prompt": (
                    "1. Is the investment in the form of a bond, note or debenture? \n"
                    "2. Does the bond, note or debenture have a maturity date or redemption date? \n"
                    "3. Does the bond, note or debenture have a repayment schedule or amortization schedule? \n"
                    "4. Does the bond, note or debenture have an interest rate, coupon rate, margin or spread? \n"
                    "5. Is the bond, note or debenture issued in series, classes, units, integral multiple or in any other denominations? \n"
                    "6. Is the bond/note or debenture issued in registered, bearer or dematerialized form? \n"
                    "7. Does the company maintain a register of bond holders or note holders or debenture holders? \n"
                    "8. Does the company issue any certificate to the bond holders or note holders or debenture holders? \n"
                    "9. Are the bonds, notes or debentures listed? \n"
                ),
            },
        ]
    },
    "Review": {
        "Financial Statement Review": [
            {
                "label": "Comprehensive Financial Statement Validation",
                "explanation": "Comprehensive checklist covering numeric formatting, currency conventions, terminology accuracy, calculation verification, and consistency checks.",
                "prompt": """1) Verify that all billion values include a decimal point (e.g., '1.0 billion' not '1 billion')
2) Check that all currency references in paragraph text use proper case sensitivity such that only the first letter of the country name is capitalized and the name of the currency itself is not capitalised (e.g., 'Indian rupee' not 'Indian Rupee' or 'indian rupee')
3) Identify potential word confusion errors such as 'principal' instead of 'principle' or 'affect' instead of 'effect'
4) Check that there are no placeholder values e.g. 0 million, 0.0 billion, 0% etc""",
            },
        ]
    },
}