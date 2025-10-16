import os
import streamlit as st
import fitz  # PyMuPDF
import openai
import re
import json
import asyncio
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import logging
import functools
import time


# --- Small UI helpers ---
def apply_ui_styling():
    """Inject lightweight CSS to give the app a cleaner, branded look."""
    css = """
    <style>
    /* Header/banner */
    .header-banner { background: linear-gradient(90deg,#f7fbff,#eef6ff); padding:10px 12px; border-radius:8px; margin-bottom:8px; }
    .brand-small { font-weight:700; color:#0b56d6; font-size:1.05rem; }
    /* Card-like containers */
    .stContainer { padding: 6px; }
    .card { border: 1px solid #e6ebf2; padding: 10px; border-radius: 8px; background: #ffffff; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_branding():
    """Render compact branding in the sidebar"""
    with st.sidebar:
        st.markdown("<div class='header-banner'><span class='brand-small'>SmartReview</span></div>", unsafe_allow_html=True)
        st.markdown("Powered by CNT")
        st.markdown("---")

# --- Logging setup and helpers ---
LOGGER_NAME = "app.py"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    # Configure basic logging to the terminal
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def _safe_str(obj, max_len=500):
    """Return a safe, truncated string representation for logging."""
    try:
        s = str(obj)
    except Exception:
        s = repr(obj)
    if len(s) > max_len:
        return s[:max_len] + "... [truncated]"
    return s


def log_sync(func):
    """Decorator for logging entry/exit/exceptions for sync functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug("ENTER %s - args=%s kwargs=%s", func.__name__, _safe_str(args), _safe_str(kwargs))
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.debug("EXIT %s - duration=%.3fs result=%s", func.__name__, duration, _safe_str(result))
            return result
        except Exception as e:
            duration = time.time() - start
            logger.exception("EXCEPTION in %s after %.3fs: %s", func.__name__, duration, e)
            raise
    return wrapper


def log_async(func):
    """Decorator for logging entry/exit/exceptions for async functions."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger.debug("ENTER async %s - args=%s kwargs=%s", func.__name__, _safe_str(args), _safe_str(kwargs))
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.debug("EXIT async %s - duration=%.3fs result=%s", func.__name__, duration, _safe_str(result))
            return result
        except Exception as e:
            duration = time.time() - start
            logger.exception("EXCEPTION in async %s after %.3fs: %s", func.__name__, duration, e)
            raise
    return wrapper


# --- LLM Client Initialization (OpenAI or Databricks) ---
# By default we use OpenAI. To enable Databricks LLM, set the following in your config or environment:
# - USE_DATABRICKS_LLM = True (in your config module or environment)
# - DATABRICKS_API_KEY env var must contain a valid Databricks token
# - DATABRICKS_BASE_URL can be overridden via env var if needed

# Default Databricks config (can be overridden by env)
DATABRICKS_BASE_URL = os.environ.get("DATABRICKS_BASE_URL", "https://adb-3858882779799477.17.azuredatabricks.net/serving-endpoints")
DATABRICKS_LLM_MODEL = os.environ.get("DATABRICKS_LLM_MODEL", "databricks-llama-4-maverick")


@st.cache_resource
def get_llm_client():
    """
    Initialize an OpenAI-compatible client configured to talk to Databricks Serving Endpoints.

    This application uses Databricks as the single, default LLM provider.
    """
    try:
        logger.info("Initializing Databricks LLM client")

        # Primary: environment variable
        databricks_token = os.environ.get("DATABRICKS_API_KEY")

        # Fallback: Streamlit secrets (useful during local dev / deployed Streamlit)
        try:
            if (not databricks_token) and hasattr(st, "secrets") and "DATABRICKS_API_KEY" in st.secrets:
                databricks_token = st.secrets["DATABRICKS_API_KEY"]
                logger.info("Loaded DATABRICKS_API_KEY from Streamlit secrets")
        except Exception:
            # Safe to ignore access to secrets failing
            pass

        if not databricks_token:
            st.error("Databricks API token not found. Please add DATABRICKS_API_KEY to your environment or Streamlit secrets.")
            st.stop()

        # Clean token (strip whitespace and any surrounding quotes)
        if isinstance(databricks_token, str):
            clean_token = databricks_token.strip().strip('"').strip("'")
        else:
            clean_token = str(databricks_token)

        # Expose Databricks creds to the process so other modules (evaluator) can use them
        try:
            os.environ["DATABRICKS_API_KEY"] = clean_token
            os.environ["DATABRICKS_BASE_URL"] = DATABRICKS_BASE_URL
            os.environ["DATABRICKS_LLM_MODEL"] = DATABRICKS_LLM_MODEL
        except Exception:
            pass

        # Try to construct the OpenAI-compatible client. Some versions of the OpenAI
        # library may ignore the api_key parameter and require the OPENAI_API_KEY env var,
        # so we attempt both approaches for compatibility.
        try:
            client = openai.OpenAI(api_key=clean_token, base_url=DATABRICKS_BASE_URL)
            client._default_model = DATABRICKS_LLM_MODEL
            logger.info("Databricks OpenAI-compatible client initialized via direct api_key")
            return client
        except Exception as first_err:
            logger.warning("Direct OpenAI(...) init failed, will attempt env-var fallback: %s", first_err)

            # Set OPENAI_API_KEY env var as a fallback and try again
            try:
                os.environ["OPENAI_API_KEY"] = clean_token
                # Also try to set module-level attribute if present
                try:
                    setattr(openai, "api_key", clean_token)
                except Exception:
                    pass

                client = openai.OpenAI(base_url=DATABRICKS_BASE_URL)
                client._default_model = DATABRICKS_LLM_MODEL
                logger.info("Databricks OpenAI-compatible client initialized via OPENAI_API_KEY env var")
                return client
            except Exception as second_err:
                # Both approaches failed; raise a helpful error
                logger.exception("Failed to initialize Databricks LLM client after both direct and env-var attempts: %s; %s", first_err, second_err)
                st.error(
                    "Failed to initialize Databricks LLM client. Ensure DATABRICKS_API_KEY is a valid token and matches your installed OpenAI client expectations."
                )
                st.stop()
    except Exception as e:
        logger.exception("Failed to initialize Databricks LLM client (unexpected): %s", e)
        st.error(f"Failed to initialize Databricks LLM client: {e}")
        st.stop()


# Create the Databricks-only client
client = get_llm_client()


async def _chat_completion_async(messages, model: Optional[str] = None, temperature: Optional[float] = None):
    """Run the (synchronous) client.chat.completions.create in a thread executor so it can be awaited.

    This makes the Databricks OpenAI-compatible client usable from async code.
    """
    model = model or getattr(client, "_default_model", DATABRICKS_LLM_MODEL)

    def _sync_call():
        kwargs = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return client.chat.completions.create(**kwargs)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_call)


def _parse_model_json(text: str) -> dict:
    """Attempt to extract JSON from a model output string.

    Strategies:
    - Try direct json.loads
    - Strip fenced code blocks (```json ... ``` or ``` ... ```)
    - Find the first {...} JSON object in the text and parse that

    Raises JSONDecodeError if parsing fails.
    """
    if not text or not text.strip():
        raise json.JSONDecodeError("Empty model response", text, 0)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stripped = text.strip()

    # If the model wrapped the JSON in fenced code blocks (e.g. ```json ... ```),
    # try to extract any fenced blocks and parse their contents.
    fence_blocks = []
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    for m in fence_pattern.finditer(stripped):
        fence_blocks.append(m.group(1).strip())

    for block in fence_blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            # try to be forgiving: sometimes models put plain text before/after
            # the JSON inside the fence
            start = block.find('{')
            end = block.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = block[start:end+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Try to locate the first '{' and the last '}' and parse that slice.
    # This is more reliable than regex for complex nested JSON with escaped characters.
    start = stripped.find('{')
    end = stripped.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error on candidate (first {{ to last }}): {e}")
            pass

    # If that fails, try to find the first JSON object or array using regex (less reliable for complex JSON).
    # Note: regex cannot fully validate nested JSON but works for simple model outputs.
    obj_pattern = re.compile(r"(\{(?:[^{}]|\{[^}]*\})*\})", re.DOTALL)
    arr_pattern = re.compile(r"(\[(?:[^\[\]]|\[[^\]]*\])*\])", re.DOTALL)

    for pattern in (obj_pattern, arr_pattern):
        for m in pattern.finditer(stripped):
            candidate = m.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # try next match
                continue

    # If all attempts fail, raise with a helpful message including a short snippet
    # of the model output to aid debugging.
    snippet = (stripped[:1000] + '...') if len(stripped) > 1000 else stripped
    raise json.JSONDecodeError(f"Could not parse JSON from model output. Snippet: {snippet}", text, 0)


# --- Pydantic Models for Structured Data ---

class ProposedValidation(BaseModel):
    """AI's proposal for how to validate a user-defined rule."""
    explanation: str = Field(..., description="A clear, user-friendly explanation of how the validation will work.")
    validation_type: Literal['regex', 'semantic'] = Field(..., description="The type of validation method the AI has chosen.")
    validator: str = Field(..., description="The generated regex pattern or the detailed semantic prompt for the LLM.")
    example_finding: str = Field(..., description="A direct quote from the provided document text that this validator would find, to show the user it works as expected.")

class Rule(BaseModel):
    """A user-confirmed, executable validation rule."""
    description: str
    validation_type: Literal['regex', 'semantic']
    validator: str

class ValidationTemplate(BaseModel):
    """A collection of rules saved by the user."""
    name: str
    rules: List[Rule]

class ValidationResult(BaseModel):
    """Represents a single issue found during validation."""
    page_num: int
    rule_description: str
    violation_type: Optional[str] = None
    finding: str
    analysis: Optional[str] = None
    context: str # A snippet of text around the finding for context

class DocumentChunk(BaseModel):
    """Represents a chunk of text from the document, typically a page."""
    content: str
    page_num: int


# --- Core Logic Functions ---
@log_sync
def decompose_rule_smartreview(rule_text: str) -> List[Rule]:
    """Decompose a plain-text rule into SmartReview validation tasks.
    - No RAG/retrieval. Single-step analysis to choose regex vs semantic.
    - Integrates validation type determination into decomposition.
    - Returns a list of Rule objects suitable for the SmartReview pipeline.
    """
    text = (rule_text or "").strip().lower()
    tasks: List[Rule] = []

    # Heuristics: precise formats → semantic with guardrails; qualitative/intent/tone → semantic
    # NOTE: We no longer hardcode regex patterns. Instead, use propose_validation_from_rule()
    # which has an AI agent that can generate optimal regex patterns with proper lookahead/lookbehind.
    format_keywords = [
        "yyyy", "mm", "dd", "date", "currency", "usd", "$", "eur", "€",
        "email", "e-mail", "phone", "ssn", "id", "identifier", "format",
        "digits", "characters", "alphanumeric", "exactly", "pattern",
    ]
    if any(k in text for k in format_keywords):
        # Default to semantic when we cannot synthesize a robust regex deterministically here.
        # The user can refine this into a regex later in the UI if desired.
        semantic_prompt = (
            f"You must check the text for violations of this rule:\n\"{rule_text}\"\n"
            "- Quote the exact offending text if any.\n"
            "- Provide a concise reason for the violation.\n"
            "- Do NOT flag compliant cases; only clear violations.\n"
            "- Be strict about numeric/word boundaries; avoid partial matches.\n"
            "- If the rule requires decimal precision, do NOT flag numbers that already include a decimal fraction (e.g., '67.3 billion', '1.0 billion').\n"
        )
        tasks.append(Rule(description=rule_text, validation_type='semantic', validator=semantic_prompt))
        return tasks

    # Default: semantic validation
    default_prompt = (
        f"You must check the text for violations of this rule:\n\"{rule_text}\"\n"
        "- Quote the exact offending text if any.\n"
        "- Provide a concise reason for the violation.\n"
        "- Do NOT flag compliant cases.\n"
        "- Be strict about boundaries; avoid partial/embedded matches.\n"
        "- If the rule mentions 'billion' and decimal precision, do NOT flag values with a decimal part (e.g., '67.3 billion', '1.0 billion').\n"
    )
    tasks.append(Rule(description=rule_text, validation_type='semantic', validator=default_prompt))
    return tasks


@log_sync
def extract_text_from_pdf(uploaded_file_bytes: bytes) -> List[DocumentChunk]:
    """Extracts text from each page of an uploaded PDF file."""
    logger.info("Starting PDF text extraction. bytes=%s", _safe_str(len(uploaded_file_bytes) if uploaded_file_bytes is not None else None))
    chunks = []
    try:
        pdf_document = fitz.open(stream=uploaded_file_bytes, filetype="pdf")
        for page_num, page in enumerate(pdf_document):
            text = page.get_text()
            logger.debug("Extracted page %s text length=%d", page_num + 1, len(text))
            chunks.append(DocumentChunk(content=text, page_num=page_num + 1))
        logger.info("Completed PDF extraction. pages=%d", len(chunks))
    except Exception as e:
        logger.exception("Error processing PDF file: %s", e)
        st.error(f"Error processing PDF file: {e}")
    return chunks

@log_async
async def propose_validation_from_rule(rule_text: str, example_text: str, doc_chunks: List[DocumentChunk]) -> Optional[ProposedValidation]:
    """AI agent to interpret a user's rule and propose a validation method."""

    # Combine document chunks for context, but limit the size to avoid excessive token usage
    full_text_context = "\n".join([chunk.content for chunk in doc_chunks])
    # Truncate context to a reasonable length for the API call
    max_context_length = 12000
    if len(full_text_context) > max_context_length:
        full_text_context = full_text_context[:max_context_length] + "\n... [document truncated for brevity]"

    system_prompt = """
    You are an expert AI system that converts a user's plain-text rule into a structured, machine-executable validation.

    Steps:
    1) Analyze the rule and any provided example.
    2) Choose validation_type: 'regex' (precise patterns like dates/currency/IDs) or 'semantic' (intent/tone/meaning/context).
    3) Generate validator:
       - If 'regex': produce a robust Python-compatible regex pattern string.
       - If 'semantic': produce a clear, concise evaluation prompt for another AI to check violations.
    4) Find an example from the provided document text that your validator would identify.
    5) Explain the approach in one or two sentences.

    CRITICAL REGEX GUIDELINES:
    When creating regex patterns, be aware of word boundaries and decimal numbers:

    - PROBLEM: Word boundary \\b treats '.' as a boundary, so "67.3 billion" is seen as two tokens: "67" and "3"
      If you write \\b\\d+\\s+billion\\b, it will match BOTH "67 billion" AND "3 billion" (from "67.3 billion")

    - SOLUTION: Use negative lookbehind and lookahead to prevent matching decimal parts:
      * (?<!\\.) - No decimal point immediately before the number
      * (?<!\\d\\.) - No digit-dot pattern before (prevents matching "3" in "67.3")
      * (?!\\.\\d+) - No decimal point after the number

    - EXAMPLE: To match integers like "67 billion" but NOT decimals like "67.3 billion":
      Pattern: (?<!\\.)(?<!\\d\\.)\\b(?:\\d{1,3}(?:,\\d{3})*|\\d+)(?!\\.\\d+)\\s+billion\\b

      This pattern:
      ✓ Matches: "67 billion", "1,234 billion", "5 billion"
      ✗ Does NOT match: "67.3 billion", "1.0 billion", "5.5 billion"

    - GENERAL RULE: When matching numbers that should NOT include decimals:
      1. Add (?<!\\.) and (?<!\\d\\.) before your number pattern
      2. Add (?!\\.\\d+) after your number pattern
      3. This prevents matching decimal parts as separate integers

    STRICT OUTPUT REQUIREMENTS:
    - Output MUST be a single JSON object with EXACTLY these keys:
      - "explanation": string
      - "validation_type": "regex" | "semantic"
      - "validator": string
      - "example_finding": string
    - Do not include any additional keys, prose, comments, markdown, or code fences.
    - Return ONLY the JSON object.
    """

    user_prompt = f"""
    Here is the document context I am working with:
    --- DOCUMENT TEXT ---
    {full_text_context}
    --- END DOCUMENT TEXT ---

    Please create a validation proposal for the following rule:
    Rule Description: "{rule_text}"
    User-provided Example: "{example_text if example_text else 'No example provided.'}"
    """

    logger.info("Requesting proposal for rule. rule_text=%s example_provided=%s doc_chunks=%d", _safe_str(rule_text), bool(example_text), len(doc_chunks))
    try:
        logger.debug("Calling LLM at %s model=%s", DATABRICKS_BASE_URL, getattr(client, "_default_model", "databricks-llama-4-maverick"))
        response = await _chat_completion_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=getattr(client, "_default_model", "databricks-llama-4-maverick")
        )
        raw = response.choices[0].message.content
        logger.debug("Raw AI response length=%d", len(_safe_str(raw)))
        try:
            response_json = _parse_model_json(raw)

            # Normalize validator field: the model sometimes returns a dict
            # like {"regex_pattern": "..."} or {"pattern": "..."}.
            # Ensure `validator` is a string as required by ProposedValidation.
            if isinstance(response_json, dict) and 'validator' in response_json:
                val = response_json['validator']
                if isinstance(val, dict):
                    # Common keys the model might use
                    for key in ('regex_pattern', 'pattern', 'regex', 'prompt', 'semantic_prompt'):
                        if key in val:
                            response_json['validator'] = val[key]
                            break
                    else:
                        # Fallback to JSON string representation
                        try:
                            response_json['validator'] = json.dumps(val)
                        except Exception:
                            response_json['validator'] = str(val)

            # Post-parse guard: ensure required keys exist
            required_keys = {"explanation", "validation_type", "validator", "example_finding"}
            if not isinstance(response_json, dict):
                response_json = {}
            missing = required_keys - set(response_json.keys())
            if missing:
                logger.warning("ProposedValidation missing keys %s; attempting structured retry", missing)
                retry_system_prompt = (
                    "Your previous output was missing required keys. Return ONLY a single JSON object with EXACTLY the keys: "
                    "explanation, validation_type, validator, example_finding. No prose, no markdown, no extra keys."
                )
                retry_user_prompt = f"""
                Rule: "{rule_text}"
                Example: "{example_text if example_text else 'No example provided.'}"
                Previous_output (verbatim):
                {raw}

                Document context (truncated):
                {full_text_context[:2000]}
                """
                retry_resp = await _chat_completion_async(
                    messages=[
                        {"role": "system", "content": retry_system_prompt},
                        {"role": "user", "content": retry_user_prompt},
                    ],
                    model=getattr(client, "_default_model", "databricks-llama-4-maverick"),
                    temperature=0.0,
                )
                raw2 = retry_resp.choices[0].message.content
                try:
                    response_json = _parse_model_json(raw2)
                    # Re-normalize validator if needed
                    if isinstance(response_json, dict) and 'validator' in response_json:
                        val2 = response_json['validator']
                        if isinstance(val2, dict):
                            for key in ('regex_pattern', 'pattern', 'regex', 'prompt', 'semantic_prompt'):
                                if key in val2:
                                    response_json['validator'] = val2[key]
                                    break
                            else:
                                try:
                                    response_json['validator'] = json.dumps(val2)
                                except Exception:
                                    response_json['validator'] = str(val2)
                except Exception:
                    logger.exception("Retry parse failed; will attempt fallback synthesis if possible")

                # Check again
                if not isinstance(response_json, dict):
                    response_json = {}
                missing = required_keys - set(response_json.keys())
                if missing:
                    if missing == {"explanation"}:
                        # Synthesize minimal explanation and proceed
                        vtype = response_json.get("validation_type", "semantic")
                        response_json["explanation"] = (
                            f"This {vtype} validator checks the document for: {rule_text.strip()[:200]}."
                        )
                    else:
                        raise ValueError(f"Missing required keys after retry: {missing}")

            proposal = ProposedValidation(**response_json)
        except Exception as e:
            logger.exception("Failed to parse JSON from model output: %s", e)
            logger.debug("Raw model output: %s", _safe_str(raw, max_len=2000))
            st.error("AI returned an unparsable response. Check logs for the raw output.")
            return None
        logger.info("Generated proposal type=%s example_finding=%s", proposal.validation_type, _safe_str(proposal.example_finding, max_len=200))
        return proposal
    except Exception as e:
        logger.exception("An AI communication error occurred while proposing validation: %s", e)
        st.error(f"An AI communication error occurred: {e}")
        return None

@log_async
async def refine_validation_from_chat(chat_history: List[dict], original_proposal: ProposedValidation, doc_chunks: List[DocumentChunk]) -> Optional[ProposedValidation]:
    """AI agent that refines a proposal based on user chat feedback."""
    full_text_context = "\n".join([chunk.content for chunk in doc_chunks])
    max_context_length = 12000
    if len(full_text_context) > max_context_length:
        full_text_context = full_text_context[:max_context_length] + "\n... [document truncated for brevity]"

    system_prompt = """
    You are an expert AI system that refines validation rules based on user feedback.
    The user was not satisfied with your previous proposal and will provide feedback.
    Your task is to generate a *new* `ProposedValidation` JSON object that incorporates the user's feedback.
    Carefully read the chat history to understand what the user wants to change. Then, generate a completely new proposal that addresses their concerns.

    CRITICAL REGEX GUIDELINES:
    When creating regex patterns, be aware of word boundaries and decimal numbers:

    - PROBLEM: Word boundary \\b treats '.' as a boundary, so "67.3 billion" is seen as two tokens: "67" and "3"
      If you write \\b\\d+\\s+billion\\b, it will match BOTH "67 billion" AND "3 billion" (from "67.3 billion")

    - SOLUTION: Use negative lookbehind and lookahead to prevent matching decimal parts:
      * (?<!\\.) - No decimal point immediately before the number
      * (?<!\\d\\.) - No digit-dot pattern before (prevents matching "3" in "67.3")
      * (?!\\.\\d+) - No decimal point after the number

    - EXAMPLE: To match integers like "67 billion" but NOT decimals like "67.3 billion":
      Pattern: (?<!\\.)(?<!\\d\\.)\\b(?:\\d{1,3}(?:,\\d{3})*|\\d+)(?!\\.\\d+)\\s+billion\\b

      This pattern:
      ✓ Matches: "67 billion", "1,234 billion", "5 billion"
      ✗ Does NOT match: "67.3 billion", "1.0 billion", "5.5 billion"

    - GENERAL RULE: When matching numbers that should NOT include decimals:
      1. Add (?<!\\.) and (?<!\\d\\.) before your number pattern
      2. Add (?!\\.\\d+) after your number pattern
      3. This prevents matching decimal parts as separate integers
    """

    chat_transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    user_prompt = f"""
    --- DOCUMENT TEXT ---
    {full_text_context}
    --- END DOCUMENT TEXT ---

    This was my original proposal that the user wants to refine:
    --- ORIGINAL PROPOSAL ---
    {original_proposal.model_dump_json(indent=2)}
    --- END ORIGINAL PROPOSAL ---

    Here is the conversation so far:
    --- CHAT HISTORY ---
    {chat_transcript}
    --- END CHAT HISTORY ---

    Based on the user's feedback in the chat, please generate a new and improved validation proposal in the required JSON format.
    """

    logger.info("Refining validation from chat. chat_length=%d original_type=%s", len(chat_history), original_proposal.validation_type if original_proposal else None)
    try:
        logger.debug("Calling LLM at %s model=%s", DATABRICKS_BASE_URL, getattr(client, "_default_model", "databricks-llama-4-maverick"))
        response = await _chat_completion_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=getattr(client, "_default_model", "databricks-llama-4-maverick")
        )
        raw = response.choices[0].message.content
        logger.debug("Raw AI refinement response length=%d", len(_safe_str(raw)))
        try:
            response_json = _parse_model_json(raw)

            # Normalize validator field similar to propose_validation_from_rule
            if isinstance(response_json, dict) and 'validator' in response_json:
                val = response_json['validator']
                if isinstance(val, dict):
                    for key in ('regex_pattern', 'pattern', 'regex', 'prompt', 'semantic_prompt'):
                        if key in val:
                            response_json['validator'] = val[key]
                            break
                    else:
                        try:
                            response_json['validator'] = json.dumps(val)
                        except Exception:
                            response_json['validator'] = str(val)

            proposal = ProposedValidation(**response_json)
        except Exception as e:
            # Parsing failed; attempt a forgiving fallback before giving up.
            logger.exception("Failed to parse JSON from refinement output: %s", e)
            logger.debug("Raw model output (snippet): %s", _safe_str(raw, max_len=2000))

            # Try to be forgiving: remove common leading phrases and code fences then retry
            cleaned = raw
            # Remove leading assistant/commentary lines like 'Here's a new proposal:'
            cleaned = re.sub(r"^\s*[A-Za-z\s\'\":,.-]{0,80}:?\s*\n", "", cleaned)
            # Remove any surrounding triple-backtick fences
            if cleaned.strip().startswith("```") and cleaned.strip().endswith("```"):
                inner_lines = cleaned.strip().splitlines()[1:-1]
                cleaned = "\n".join(inner_lines)

            try:
                response_json = _parse_model_json(cleaned)
                # Normalize as below
                if isinstance(response_json, dict) and 'validator' in response_json:
                    val = response_json['validator']
                    if isinstance(val, dict):
                        for key in ('regex_pattern', 'pattern', 'regex', 'prompt', 'semantic_prompt'):
                            if key in val:
                                response_json['validator'] = val[key]
                                break
                        else:
                            try:
                                response_json['validator'] = json.dumps(val)
                            except Exception:
                                response_json['validator'] = str(val)

                proposal = ProposedValidation(**response_json)
                logger.info("Refined proposal parsed after fallback. type=%s", proposal.validation_type)
                return proposal
            except Exception:
                logger.exception("Fallback parsing also failed for refinement output.")
                st.error("AI returned an unparsable refinement. Check logs for the raw output.")
                return None
        logger.info("Refined proposal generated type=%s", proposal.validation_type)
        return proposal
    except Exception as e:
        logger.exception("An AI communication error occurred during refinement: %s", e)
        st.error(f"An AI communication error occurred during refinement: {e}")
        return None


@log_async
async def execute_validation_template(template: ValidationTemplate, doc_chunks: List[DocumentChunk]) -> List[ValidationResult]:
    """Runs all rules in a template against a document and collects the results.
    Uses the new Orchestrator (parallel multi-tool + evaluator). Falls back to legacy execution on error.
    """
    logger.info(
        "Executing validation template '%s' with %d rules on %d document chunks.",
        template.name if template else None,
        len(template.rules) if template else 0,
        len(doc_chunks),
    )
    try:
        from src.keyword_code.agents.review_orchestrator import orchestrate_review
        ranked = await orchestrate_review(template, doc_chunks)
        results: List[ValidationResult] = []
        for r in ranked:
            results.append(
                ValidationResult(
                    page_num=r.page_num,
                    rule_description=r.rule_description,
                    violation_type=r.violation_type,
                    finding=r.finding,
                    analysis=r.analysis,
                    context=r.context,
                )
            )
        return results
    except Exception as e:
        logger.exception("Orchestrated review failed, falling back to legacy execution: %s", e)
        all_results: List[ValidationResult] = []
        tasks = []
        for rule in template.rules:
            for chunk in doc_chunks:
                tasks.append(run_rule_on_chunk(rule, chunk))
        list_of_results_per_task = await asyncio.gather(*tasks)
        for result_list in list_of_results_per_task:
            all_results.extend(result_list)
        return all_results

@log_async
async def run_rule_on_chunk(rule: Rule, chunk: DocumentChunk) -> List[ValidationResult]:
    """Helper function to run a single rule on a single chunk."""
    results = []
    logger.debug("Running rule on chunk. rule='%s' type=%s page=%d", _safe_str(rule.description, max_len=200), rule.validation_type, chunk.page_num)
    if rule.validation_type == 'regex':
        try:
            matches = re.finditer(rule.validator, chunk.content)
            for match in matches:
                # Create a context snippet around the finding
                start = max(0, match.start() - 50)
                end = min(len(chunk.content), match.end() + 50)
                context_snippet = f"...{chunk.content[start:end]}..."
                logger.debug("Regex match on page %d: %s", chunk.page_num, _safe_str(match.group(0), max_len=200))
                results.append(ValidationResult(
                    page_num=chunk.page_num,
                    rule_description=rule.description,
                    violation_type='regex',
                    finding=f"Found violation: '{match.group(0)}'",
                    analysis=f"Regex matched the pattern for this rule, indicating non-compliance: {rule.description}",
                    context=context_snippet
                ))
        except re.error as e:
            logger.exception("Invalid regex pattern for rule '%s': %s", rule.description, e)
            st.warning(f"Invalid regex pattern for rule '{rule.description}': {e}")

    elif rule.validation_type == 'semantic':
        system_prompt = """
        You are an AI document validation assistant. You will be given a chunk of text and a rule.
        Your task is to check if the text violates the rule.
        - If you find a violation, respond with "Violation: [Explain the violation and quote the specific text]".
        - If there are no violations, respond *only* with the text "No violation found.".
        - Do NOT flag numeric expressions that already satisfy the rule; for example, if decimal precision like "1.0 billion" is required, values like "67.3 billion" are compliant and must not be flagged; only integer forms like "67 billion" should be flagged.
        Do not be conversational. Provide only the violation report or "No violation found.".
        """
        prompt = f"""
        --- RULE ---
        {rule.validator}

        --- TEXT TO VALIDATE ---
        {chunk.content}
        """
        try:
            logger.debug("Calling semantic AI for page %d rule=%s; base_url=%s model=%s", chunk.page_num, _safe_str(rule.description, max_len=200), DATABRICKS_BASE_URL, getattr(client, "_default_model", "databricks-llama-4-maverick"))
            response = await _chat_completion_async(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=getattr(client, "_default_model", "databricks-llama-4-maverick"),
                temperature=0.1,
            )
            message_content = response.choices[0].message.content
            logger.debug("Semantic AI response for page %d length=%d", chunk.page_num, len(_safe_str(message_content)))
            if message_content.lower().strip() != "no violation found.":
                logger.info("Semantic violation found on page %d: %s", chunk.page_num, _safe_str(message_content, max_len=300))
                results.append(ValidationResult(
                    page_num=chunk.page_num,
                    rule_description=rule.description,
                    violation_type='semantic',
                    finding=message_content,
                    analysis=f"Semantic evaluation reason: {message_content}",
                    context=f"Semantic check on page {chunk.page_num}."
                ))
        except Exception as e:
            logger.exception("API call failed for semantic rule on page %d: %s", chunk.page_num, e)
            st.warning(f"API call failed for semantic rule on page {chunk.page_num}: {e}")

    return results

# --- UI Rendering Functions ---

@log_sync
def render_validation_view():
    """Renders the main UI for validating documents against saved templates."""
    logger.debug("Rendering validation view")
    # Top-level title and short description
    st.title("SmartReview — Document Validation")
    st.caption("Upload a PDF, pick a template, and run fast AI- or regex-based checks.")

    # Minimal sidebar: navigation and a small help/credits area
    with st.sidebar:
        st.markdown("### Navigation")
        if st.button("Create Template"):
            st.session_state.app_mode = 'rule_definition'
            st.rerun()
        st.markdown("---")
        st.markdown("#### Tips")
        st.write("Keep templates focused: 3-8 rules works well.")
        st.markdown("---")
        if st.checkbox("Show debug logs", value=False, key="show_logs"):
            st.write("Logs are printed to the server console.")

    # If there are no templates, show a friendly prompt
    if not st.session_state.templates:
        st.info("No validation templates yet. Click 'Create Template' in the sidebar to get started.")
        return

    # Main validation area laid out in two columns: inputs (left) and results (right)
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Run Validation")
        template_names = list(st.session_state.templates.keys())
        selected_template_name = st.selectbox("Select a Template", options=template_names)
        uploaded_file = st.file_uploader("Upload PDF to validate", type="pdf", key="validation_uploader")
        run_disabled = (not selected_template_name) or (not uploaded_file)
        if st.button("Run Validation", disabled=run_disabled, type="primary"):
            with st.spinner("Analyzing document... this may take a moment"):
                template = st.session_state.templates[selected_template_name]
                pdf_bytes = uploaded_file.getvalue()
                doc_chunks = extract_text_from_pdf(pdf_bytes)
                if doc_chunks:
                    results = asyncio.run(execute_validation_template(template, doc_chunks))
                    st.session_state.validation_results = results
                    st.success("Validation complete.")
                else:
                    st.error("Could not extract text from the PDF.")

    # Results column: shows run status and collapsible results per finding
    with right_col:
        st.subheader("Validation Report")
        if st.session_state.validation_results is None:
            st.info("No validation run yet. Results will appear here.")
        elif not st.session_state.validation_results:
            st.success("No issues were found for the selected template.")
        else:
            for result in st.session_state.validation_results:
                header = f"[{(result.violation_type or 'violation').upper()}] Page {result.page_num} — {result.rule_description}"
                with st.expander(header, expanded=False):
                    if getattr(result, 'analysis', None):
                        st.markdown(result.analysis)
                    st.error(result.finding)
                    st.caption("Context")
                    st.markdown(f"> {result.context.replace('...', ' ... ')}")

@log_sync
def render_rule_definition_view():
    """Renders the UI for the interactive rule creation process."""
    logger.debug("Rendering rule definition view")
    # Rule definition screen: upload a sample doc, then create rules with AI assistance
    st.header("Rule Template Definition")

    # Compact sidebar control to go back to validation
    with st.sidebar:
        if st.button("Back to Validation"):
            st.session_state.app_mode = 'validation'
            st.rerun()
        st.markdown("---")

    # Step 1: Upload a sample document if none exists
    if not st.session_state.definition_pdf_bytes:
        st.info("Upload a sample PDF to base your template on.")
        uploaded_file = st.file_uploader("Upload sample PDF", type="pdf", key="definition_uploader")
        if uploaded_file:
            st.session_state.definition_pdf_bytes = uploaded_file.getvalue()
            st.session_state.definition_doc_chunks = extract_text_from_pdf(st.session_state.definition_pdf_bytes)
            st.rerun()

    if not st.session_state.definition_pdf_bytes:
        return

    st.success("Sample document loaded.")

    # Layout: left column shows accepted rules, right column is for creating/confirming a new rule
    left, right = st.columns([1, 1.2])

    with right:
        st.subheader("Accepted Rules")
        if not st.session_state.current_rules:
            st.caption("No rules added yet.")
        else:
            for i, rule in enumerate(st.session_state.current_rules):
                with st.expander(f"{i+1}. {rule.description}"):
                    st.code(f"Type: {rule.validation_type}\nValidator: {rule.validator}", language="text")

    with left:
        st.subheader("Create a New Rule")

        # If AI has proposed a validation, show a compact confirmation card
        if st.session_state.proposed_validation:
            proposal = st.session_state.proposed_validation
            st.markdown("**AI Proposal**")
            st.markdown(f"**Explanation:** {proposal.explanation}")
            st.markdown(f"**Type:** `{proposal.validation_type}`")
            st.code(proposal.validator, language='text')
            st.markdown(f"**Example finding:** {proposal.example_finding}")

            a_col, b_col = st.columns([1, 1])
            with a_col:
                if st.button("Accept Proposal"):
                    accepted_rule = Rule(
                        description=st.session_state.current_rule_text,
                        validation_type=proposal.validation_type,
                        validator=proposal.validator,
                    )
                    st.session_state.current_rules.append(accepted_rule)
                    st.session_state.proposed_validation = None
                    st.session_state.current_rule_text = ""
                    st.session_state.current_rule_example = ""
                    st.session_state.refinement_chat_history = []
                    st.rerun()
            with b_col:
                if st.button("Refine"):
                    st.session_state.is_refining = True
                    st.session_state.refinement_chat_history.append({"role": "assistant", "content": "I've created a proposal. How would you like to refine it?"})
                    st.rerun()

        # Refinement chat
        if st.session_state.is_refining:
            with st.expander("Refine with AI", expanded=True):
                for message in st.session_state.refinement_chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                if prompt := st.chat_input("Tell the AI how to change the rule..."):
                    st.session_state.refinement_chat_history.append({"role": "user", "content": prompt})
                    with st.spinner("AI is thinking..."):
                        new_proposal = asyncio.run(refine_validation_from_chat(
                            st.session_state.refinement_chat_history,
                            st.session_state.proposed_validation,
                            st.session_state.definition_doc_chunks,
                        ))
                        if new_proposal:
                            st.session_state.proposed_validation = new_proposal
                            st.session_state.refinement_chat_history.append({"role": "assistant", "content": "Based on your feedback, here is my new proposal."})
                        else:
                            st.session_state.refinement_chat_history.append({"role": "assistant", "content": "I had trouble processing that refinement. Could you rephrase?"})
                    st.rerun()

        # Main form for new rules (shown when not confirming/refining)
        if not st.session_state.proposed_validation and not st.session_state.is_refining:
            rule_text = st.text_input("Rule (plain text)", placeholder="e.g., Dates must be YYYY-MM-DD", key="rule_text_input")
            example_text = st.text_input("Example from document (optional)", placeholder="e.g., 2025-10-27", key="rule_example_input")
            if st.button("Get AI Suggestion", disabled=not rule_text):
                st.session_state.current_rule_text = rule_text
                st.session_state.current_rule_example = example_text
                with st.spinner("AI is analyzing your rule..."):
                    proposal = asyncio.run(propose_validation_from_rule(rule_text, example_text, st.session_state.definition_doc_chunks))
                    if proposal:
                        st.session_state.proposed_validation = proposal
                st.rerun()

        # Save template area
        st.markdown("---")
        template_name = st.text_input("Template name", placeholder="e.g., Financial Report Standard")
        if st.button("Save Template", disabled=(not st.session_state.current_rules or not template_name)):
            normalized_rules = []
            for r in st.session_state.current_rules:
                try:
                    if hasattr(r, 'model_dump'):
                        normalized_rules.append(Rule(**r.model_dump()))
                    elif isinstance(r, dict):
                        normalized_rules.append(Rule(**r))
                    else:
                        normalized_rules.append(Rule(description=getattr(r, 'description', str(r)), validation_type=getattr(r, 'validation_type', 'semantic'), validator=getattr(r, 'validator', '')))
                except Exception:
                    # skip invalid rule
                    pass
            new_template = ValidationTemplate(name=template_name, rules=normalized_rules)
            st.session_state.templates[template_name] = new_template
            st.session_state.definition_pdf_bytes = None
            st.session_state.definition_doc_chunks = []
            st.session_state.current_rules = []
            st.success(f"Template '{template_name}' saved!")
            st.balloons()
            st.session_state.app_mode = 'validation'
            st.rerun()


# --- Main App Logic ---

def initialize_session_state():
    """Initializes all required keys in Streamlit's session state."""
    logger.debug("Initializing session state")
    state_defaults = {
        'app_mode': 'validation', # 'validation' or 'rule_definition'
        'templates': {},
        'validation_results': None,
        # State for rule definition view
        'definition_pdf_bytes': None,
        'definition_doc_chunks': [],
        'current_rules': [],
        'current_rule_text': "",
        'current_rule_example': "",
        'proposed_validation': None,
        'is_refining': False,
        'refinement_chat_history': []
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    logger.info("Starting Streamlit app main")
    # Set page config to match template cues (wide layout, collapsed sidebar)
    st.set_page_config(layout="wide", page_title="SmartReview - Document Intelligence", initial_sidebar_state="collapsed")
    st.markdown("<h1 style='text-align: center;'>SmartReview</h1>", unsafe_allow_html=True)
    # Small CSS refresh for a cleaner, modern look
    st.markdown(
        """
        <style>
        /* Tighten up spacing and use a neutral font-size for captions */
        .stApp { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
        .stButton>button { border-radius: 6px; }
        .stCaption { font-size: 0.9rem; color: #6c757d; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Apply the template-like styling and branding
    try:
        apply_ui_styling()
        render_branding()
    except Exception:
        # Styling/branding should never block the app; swallow errors
        logger.debug("Branding/styling helpers failed to render, continuing without them.")

    initialize_session_state()

    if st.session_state.app_mode == 'validation':
        render_validation_view()
    elif st.session_state.app_mode == 'rule_definition':
        render_rule_definition_view()

if __name__ == "__main__":
    main()