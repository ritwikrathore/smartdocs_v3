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
ENABLE_INTERACTION_LOGGING = True  # Disabled by default

# --- Highlight Debug Logging Configuration ---
# Set to True to enable detailed logging of phrase matching attempts for highlight debugging
ENABLE_HIGHLIGHT_DEBUG_LOGGING = True  # Disabled by default

# Create highlight debug logger if enabled
highlight_debug_logger = None
if ENABLE_HIGHLIGHT_DEBUG_LOGGING:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    highlight_debug_log_file = logs_dir / f"highlight_debug_{timestamp}.log"
    highlight_debug_logger = logging.getLogger("highlight_debug")
    highlight_debug_logger.setLevel(logging.DEBUG)
    highlight_debug_logger.propagate = False  # Don't propagate to root logger
    highlight_debug_handler = logging.FileHandler(highlight_debug_log_file, mode='a', encoding='utf-8')
    highlight_debug_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    highlight_debug_logger.addHandler(highlight_debug_handler)
    logger.info(f"Highlight debug logging enabled: {highlight_debug_log_file}")
else:
    highlight_debug_logger = None

# --- Databricks Models ---
# Configuration for Databricks services
USE_DATABRICKS_EMBEDDING = True  # Use Databricks for embeddings
USE_DATABRICKS_LLM = True  # Use Databricks for LLM
USE_DATABRICKS_RERANKER = False  # Use Databricks for reranking

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
                    "3. What is the loan principal amount and currency associated with the said amount for different tranches and loan types? \n"
                    "4. What is the spread rate or margin rate for different loans? \n"
                    "5. Is there a reference to a Credit Adjustment Spread (CAS)?\n"
                    "6. What are the business day definitions? \n"
                    "7. What is the applicable business day convention for adjusting the interest payment date if the scheduled date falls on a non-business day? \n"
                    "8. What are the interest payment dates? \n"
                    "9. What are the interest terms, variable or fixed rate? Is it Term SOFR, NON-USD Index, or NCCR for different loans? \n"
                    "10. What rounding adjustments, if any, are required for the interest rates? \n"
                    "11. Interest shall accrue from day to day on what basis? \n"
                    "12. What are the terms for partial prepayment / prepayment premium and allocation of principal amounts outstanding? \n"
                    "13. What is the method for applying prepayments—is it pro rata basis or in inverse order of maturity? \n"
                    "14. What are the repayment terms and can you list the full repayment schedule with dates in a table? \n"
                    "15. What are all the fees the borrower shall pay and the percentages / amounts and are there any references of a separate fee letter? \n"
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
            {
                "label": "Loan Option Analysis",
                "prompt": (
                    "1. What is the loan/facility amount? \n"
                    "2. What is the interest rate/margin/spread for this loan? \n"
                    "3. Is there any interest rate step up/ interest rate step down/ interest rate caps or interest rate floors on the loan? \n"
                    "4. What is the default rate interest/default interest rate of this loan? \n"
                    "5. Is there any - No term SOFR recommended fallback rate or Term recommended fallback rate index cessation effective date provision in the agreement? \n"
                    "6. What is the interest rate for market disruption/ market disruption event/benchmark replacement event/replacement of benchmark rate? \n"
                    "7. What is the repayment/redemption provision of the loan? \n"
                    "8. Does the loan contain any extension/rollover provision? \n"
                    "9. Can the borrower voluntarily prepay the loan? \n"
                    "10. What are the conditions that would require the borrower to make an early loan repayment/prepayment? \n"
                    "11. Does the loan contain any mandatory prepayment/redemption or acceleration provisions? \n"
                    "12. Is there any prepayment premium? \n"
                    "13. What constitutes fees, increased costs and unwinding costs in the loan agreement? \n"
                    "14. Does the agreement contain a make-whole amount provision? \n"
                    "15. Is the loan due and payable if the borrower is liquidated or declared bankrupt? \n"
                    "16. Is there any illegality or illegality of participation provision in the agreement? \n"
                    "17. Are the borrowers jointly and severally liable for the loan? \n"
                    "18. Is there any bail-in provision, loss absorption provision or subordination provision in the agreement? \n"
                ),
            },
            {
                "label": "Guarantee Analysis",
                "prompt": (
                    "1. What constitutes credit event, loss, proof of loss, covered loss, net loss, borrower default, event of default? \n"
                    "2. Is IFC entitled to a right to recoveries in respect of the covered loss? \n"
                    "3. Does any of the credit event, loss, proof of loss, covered loss, net loss, borrower default, event of default encompasses more than failure to pay? \n"
                    "4. Does borrower default mean that the borrower fails to pay the required amount on the due date? \n"
                    "5. Is there a defined grace period prior to a claim being made? \n"
                    "6. Is the guaranteed obligation/ eligible obligation/ reimbursement obligation/reference portfolio a loan/note/facility/bond? \n"
                    "7. Is IFC obligated to pay the claim amount only if, and to the extent that, the Donor/Commission provides IFC with the necessary funds? \n"
                    "8. What constitutes covered amount or covered percentage by IFC? \n"
                    "9. Under what circumstances can a claim, payment demand, default notice be made to IFC? \n"
                    "10. Does IFC agree to extend to the Borrower, stand-by Dollar loans (the \"IFC Stand-by Loan\") to finance the payment of the Obligations? \n"
                    "11. Is there any legal requirement by the guaranteed party for legal transfer of title of the covered portion of the bond holding to IFC? \n"
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