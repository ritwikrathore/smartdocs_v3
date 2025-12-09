"""
Feedback logging utility for capturing user satisfaction ratings on analysis results.

JSON Structure:
{
  "sessions": {
    "session_id_1": {
      "documents": {
        "document1.pdf": {
          "first_seen": "2025-12-06T13:00:00",
          "total_questions": 5,
          "questions": {
            "section_key_1": {
              "timestamp": "2025-12-06T13:00:00",
              "section_title": "Document Date",
              "question": "What is the document date?",
              "answer": "The document date is...",
              "citations": [
                {"text": "...", "page": 1}
              ],
              "feedback": {
                "feedback_type": "positive/negative",
                "timestamp": "2025-12-06T13:05:00",
                "additional_feedback": "..." (optional)
              }
            }
          }
        }
      }
    }
  }
}
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def get_feedback_dir() -> Path:
    """Get or create the feedback directory in the project root."""
    from ..config import root_dir
    feedback_dir = root_dir / "feedback"
    feedback_dir.mkdir(exist_ok=True)
    return feedback_dir


def _load_feedback_data() -> Dict[str, Any]:
    """Load hierarchical feedback data from JSON file."""
    feedback_dir = get_feedback_dir()
    feedback_file = feedback_dir / "feedback_log.json"

    if feedback_file.exists():
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Migrate old flat format to new hierarchical format if needed
                if isinstance(data, list):
                    logger.info("Migrating feedback data from flat to hierarchical format")
                    return {"sessions": {}}
                return data
        except json.JSONDecodeError:
            logger.warning("Feedback file is corrupted. Creating new structure.")
            return {"sessions": {}}

    return {"sessions": {}}


def _save_feedback_data(data: Dict[str, Any]) -> None:
    """Save hierarchical feedback data to JSON file."""
    feedback_dir = get_feedback_dir()
    feedback_file = feedback_dir / "feedback_log.json"

    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_question_answer(
    session_id: str,
    filename: str,
    section_key: str,
    section_title: str,
    question: str,
    answer: str,
    citations: List[Dict[str, Any]],
) -> None:
    """
    Log a question and its answer for a document.
    Call this when displaying each section in the analysis.

    Args:
        session_id: Streamlit session ID
        filename: Name of the document being analyzed
        section_key: Internal section identifier
        section_title: Display title of the section
        question: The decomposed prompt question
        answer: The AI-generated answer
        citations: List of citation dictionaries with 'text' and 'page' fields
    """
    try:
        data = _load_feedback_data()

        # Ensure session exists
        if session_id not in data["sessions"]:
            data["sessions"][session_id] = {"documents": {}}

        # Ensure document exists
        if filename not in data["sessions"][session_id]["documents"]:
            data["sessions"][session_id]["documents"][filename] = {
                "first_seen": datetime.now().isoformat(),
                "total_questions": 0,
                "questions": {}
            }

        # Ensure questions key exists
        if "questions" not in data["sessions"][session_id]["documents"][filename]:
            data["sessions"][session_id]["documents"][filename]["questions"] = {}

        # Add or update question entry (preserve existing feedback if it exists)
        existing_question = data["sessions"][session_id]["documents"][filename]["questions"].get(section_key, {})
        existing_feedback = existing_question.get("feedback")

        data["sessions"][session_id]["documents"][filename]["questions"][section_key] = {
            "timestamp": datetime.now().isoformat(),
            "section_title": section_title,
            "question": question,
            "answer": answer,
            "citations": [
                {
                    "text": citation.get("text", ""),
                    "page": citation.get("page", ""),
                }
                for citation in citations
            ]
        }

        # Restore existing feedback if it was there
        if existing_feedback:
            data["sessions"][session_id]["documents"][filename]["questions"][section_key]["feedback"] = existing_feedback

        # Update total_questions count
        total_questions = len(data["sessions"][session_id]["documents"][filename]["questions"])
        data["sessions"][session_id]["documents"][filename]["total_questions"] = total_questions

        _save_feedback_data(data)
        logger.info(f"Logged question '{section_title}' (key: {section_key}) for '{filename}' in session {session_id[:8]}...")

    except Exception as e:
        logger.error(f"Failed to log question answer: {e}", exc_info=True)


def log_feedback(
    feedback_type: str,
    filename: str,
    section_key: str,
    section_title: str,
    question: str,
    answer: str,
    citations: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    additional_feedback: Optional[str] = None,
) -> None:
    """
    Add feedback to an existing question entry in the hierarchical structure.

    Args:
        feedback_type: "positive" or "negative"
        filename: Name of the document being analyzed
        section_key: Internal section key
        section_title: Display title of the section
        question: The decomposed prompt question (kept for compatibility)
        answer: The AI-generated answer (kept for compatibility)
        citations: List of citation dictionaries (kept for compatibility)
        session_id: Optional Streamlit session ID for tracking consistency across sessions
        additional_feedback: Optional text explanation for negative feedback
    """
    try:
        if not session_id:
            logger.warning("No session_id provided for feedback logging")
            return

        data = _load_feedback_data()

        # Ensure session exists
        if session_id not in data["sessions"]:
            logger.warning(f"Session '{session_id}' not found when logging feedback - creating it")
            data["sessions"][session_id] = {"documents": {}}

        # Ensure document exists
        if filename not in data["sessions"][session_id]["documents"]:
            logger.warning(f"Document '{filename}' not found in session '{session_id}' - creating it")
            data["sessions"][session_id]["documents"][filename] = {
                "first_seen": datetime.now().isoformat(),
                "total_questions": 0,
                "questions": {}
            }

        # Ensure questions key exists
        if "questions" not in data["sessions"][session_id]["documents"][filename]:
            data["sessions"][session_id]["documents"][filename]["questions"] = {}

        # Check if question exists
        questions = data["sessions"][session_id]["documents"][filename]["questions"]
        if section_key not in questions:
            logger.warning(f"Question '{section_key}' not found for document '{filename}' - creating it as fallback")
            questions[section_key] = {
                "timestamp": datetime.now().isoformat(),
                "section_title": section_title,
                "question": question,
                "answer": answer,
                "citations": [
                    {
                        "text": citation.get("text", ""),
                        "page": citation.get("page", ""),
                    }
                    for citation in citations
                ]
            }

        # Add feedback to the question
        feedback_entry = {
            "feedback_type": feedback_type,
            "timestamp": datetime.now().isoformat(),
        }

        if additional_feedback:
            feedback_entry["additional_feedback"] = additional_feedback

        data["sessions"][session_id]["documents"][filename]["questions"][section_key]["feedback"] = feedback_entry

        # Update total_questions count
        total_questions = len(data["sessions"][session_id]["documents"][filename]["questions"])
        data["sessions"][session_id]["documents"][filename]["total_questions"] = total_questions

        _save_feedback_data(data)
        logger.info(f"Logged {feedback_type} feedback for section '{section_title}' in {filename}")

    except Exception as e:
        logger.error(f"Failed to log feedback: {e}", exc_info=True)


def append_additional_feedback(
    session_id: str,
    section_key: str,
    additional_feedback: str,
) -> bool:
    """
    Append additional feedback text to an existing question's feedback entry.

    Args:
        session_id: Streamlit session ID
        section_key: Internal section identifier
        additional_feedback: Additional text feedback from user

    Returns:
        True if successfully appended, False otherwise
    """
    try:
        data = _load_feedback_data()

        # Check if session exists
        if session_id not in data["sessions"]:
            logger.warning(f"Session '{session_id}' not found")
            return False

        # Search all documents in this session for the matching question
        for filename, doc_data in data["sessions"][session_id]["documents"].items():
            questions = doc_data.get("questions", {})

            if section_key in questions:
                question_entry = questions[section_key]

                # Check if feedback exists and is negative
                if "feedback" in question_entry:
                    if question_entry["feedback"].get("feedback_type") == "negative":
                        question_entry["feedback"]["additional_feedback"] = additional_feedback

                        _save_feedback_data(data)
                        logger.info(f"Appended additional feedback for section '{section_key}' in {filename}")
                        return True

        logger.warning(f"No matching negative feedback found for section '{section_key}'")
        return False

    except Exception as e:
        logger.error(f"Failed to append additional feedback: {e}")
        return False


def get_feedback_stats() -> Dict[str, Any]:
    """
    Get statistics about collected feedback across all sessions and documents.

    Returns:
        Dictionary with feedback statistics including total questions and feedback counts
    """
    try:
        data = _load_feedback_data()

        total_questions = 0
        total_feedback = 0
        positive = 0
        negative = 0
        sessions_count = len(data.get("sessions", {}))
        documents_count = 0

        for session_id, session_data in data.get("sessions", {}).items():
            for filename, doc_data in session_data.get("documents", {}).items():
                documents_count += 1

                questions = doc_data.get("questions", {})
                total_questions += len(questions)

                # Count feedback within questions
                for section_key, question_data in questions.items():
                    if "feedback" in question_data:
                        total_feedback += 1
                        feedback_type = question_data["feedback"].get("feedback_type")
                        if feedback_type == "positive":
                            positive += 1
                        elif feedback_type == "negative":
                            negative += 1

        return {
            "sessions": sessions_count,
            "documents": documents_count,
            "total_questions": total_questions,
            "total_feedback": total_feedback,
            "positive": positive,
            "negative": negative,
        }
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        return {
            "sessions": 0,
            "documents": 0,
            "total_questions": 0,
            "total_feedback": 0,
            "positive": 0,
            "negative": 0,
        }
