"""Keyword mode service for deterministic keyword-only retrieval and highlighting."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import fitz  # PyMuPDF

from ..config import logger
from ..processors.pdf_processor import PDFProcessor
from ..rag.retrieval import get_bm25_results


@dataclass
class KeywordOccurrence:
    """Represents a single keyword match found in the document."""

    id: str
    index: int
    keyword: str
    page_num: int
    match_score: float
    method: str
    rect: List[float]
    chunk_id: Optional[int]
    snippet: str
    positions: List[Dict[str, int]] = field(default_factory=list)

    @property
    def page_label(self) -> int:
        return self.page_num + 1 if isinstance(self.page_num, int) else self.page_num


class KeywordModeService:
    """Encapsulates the keyword-only workflow, bypassing full RAG + analysis."""

    def run(
        self,
        *,
        filename: str,
        keywords: Sequence[str],
        chunks: List[Dict[str, Any]],
        original_pdf_bytes: bytes,
    ) -> Dict[str, Any]:
        """Execute keyword mode pipeline once keywords are supplied by decomposition."""

        normalized_keywords = [kw.strip() for kw in keywords or [] if isinstance(kw, str) and kw.strip()]
        if not normalized_keywords:
            logger.error("Keyword mode invoked without explicit keywords from decomposition.")
            raise ValueError("Keyword mode requires at least one keyword.")

        logger.info("Keyword mode: processing keywords %s", normalized_keywords)

        processor = PDFProcessor(original_pdf_bytes)
        phrase_locations: Dict[str, List[Dict[str, Any]]] = {}
        verification_results: Dict[str, Dict[str, Any]] = {}
        keyword_sections: Dict[str, Dict[str, Any]] = {}
        total_occurrences = 0

        doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")
        try:
            for idx, keyword in enumerate(normalized_keywords, start=1):
                section_key = f"section_{idx}_{self._slugify(keyword)}"
                occurrences = self._find_occurrences_for_keyword(keyword, section_key, chunks, doc)

                keyword_sections[section_key] = {
                    "keyword": keyword,
                    "count": len(occurrences),
                    "occurrences": [
                        {
                            "id": occ.id,
                            "index": occ.index,
                            "keyword": occ.keyword,
                            "page_num": occ.page_num,
                            "page_label": occ.page_label,
                            "match_score": occ.match_score,
                            "method": occ.method,
                            "rect": occ.rect,
                            "chunk_id": occ.chunk_id,
                            "snippet": occ.snippet,
                            "positions": occ.positions,
                        }
                        for occ in occurrences
                    ],
                }

                for occ in occurrences:
                    phrase_locations[occ.id] = [
                        {
                            "page_num": occ.page_num,
                            "rect": occ.rect,
                            "chunk_id": occ.chunk_id,
                            "match_score": occ.match_score,
                            "method": occ.method,
                        }
                    ]
                    verification_results[occ.id] = {
                        "verified": True,
                        "score": occ.match_score,
                        "method": occ.method,
                    }

                total_occurrences += len(occurrences)

        finally:
            doc.close()

        annotated_pdf_bytes = processor.add_annotations(phrase_locations)
        annotated_pdf_b64 = base64.b64encode(annotated_pdf_bytes).decode("utf-8")

        return {
            "filename": filename,
            "annotated_pdf": annotated_pdf_b64,
            "phrase_locations": phrase_locations,
            "verification_results": verification_results,
            "keyword_mode": True,
            "keyword_mode_sections": keyword_sections,
            "total_occurrences": total_occurrences,
            "keywords": normalized_keywords,
        }

    def _find_occurrences_for_keyword(
        self,
        keyword: str,
        section_key: str,
        chunks: List[Dict[str, Any]],
        doc: fitz.Document,
    ) -> List[KeywordOccurrence]:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        bm25_scores = self._score_chunks_with_keyword(keyword, chunks)
        max_score = max(bm25_scores.values(), default=0.0)

        occurrences: List[KeywordOccurrence] = []

        for chunk_index, chunk in enumerate(chunks):
            text = chunk.get("text") or ""
            if not text:
                continue

            matches = list(pattern.finditer(text))
            if not matches:
                continue

            chunk_id = chunk.get("chunk_id") if isinstance(chunk.get("chunk_id"), int) else None
            page_num = chunk.get("page_num", -1)
            rect = self._rect_from_chunk(chunk)
            rect_coords = [rect.x0, rect.y0, rect.x1, rect.y1] if rect else [0.0, 0.0, 0.0, 0.0]

            score = bm25_scores.get(chunk_index, 0.0)
            normalized_score = self._normalize_score(score, max_score)

            for match in matches:
                occurrence_index = len(occurrences) + 1
                occurrence_id = f"{section_key}_occ_{occurrence_index}"
                snippet = self._extract_snippet(
                    doc,
                    page_num=page_num,
                    rect=fitz.Rect(rect) if rect else None,
                    keyword=keyword,
                    chunk=chunk,
                    match_span=(match.start(), match.end()),
                )

                occurrences.append(
                    KeywordOccurrence(
                        id=occurrence_id,
                        index=occurrence_index,
                        keyword=keyword,
                        page_num=page_num,
                        match_score=normalized_score,
                        method="keyword_mode_bm25",
                        rect=rect_coords,
                        chunk_id=chunk_id,
                        snippet=snippet,
                        positions=[{"start": match.start(), "end": match.end()}],
                    )
                )

        logger.info(
            "Keyword '%s' produced %d occurrence(s) across %d chunk(s).",
            keyword,
            len(occurrences),
            len({occ.chunk_id for occ in occurrences if occ.chunk_id is not None}),
        )

        return occurrences

    def _score_chunks_with_keyword(self, keyword: str, chunks: List[Dict[str, Any]]) -> Dict[int, float]:
        try:
            results = get_bm25_results(keyword, chunks, top_k=len(chunks))
        except Exception as err:
            logger.error("BM25 retrieval failed for keyword '%s': %s", keyword, err, exc_info=True)
            return {}

        return {chunk_index: float(score) for chunk_index, score in results if score and score > 0}

    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        return slug or "keyword"

    @staticmethod
    def _rect_from_chunk(chunk: Dict[str, Any]) -> Optional[fitz.Rect]:
        rect: Optional[fitz.Rect] = None
        for bbox in chunk.get("bboxes", []) or []:
            try:
                current = fitz.Rect(bbox) if not isinstance(bbox, fitz.Rect) else bbox
            except Exception:  # pragma: no cover - defensive
                continue
            if rect is None:
                rect = fitz.Rect(current)
            else:
                rect.include_rect(current)
        return rect

    @staticmethod
    def _normalize_score(score: float, max_score: float) -> float:
        if max_score <= 0:
            return 0.0
        normalised = score / max_score
        return round(normalised, 4)

    @staticmethod
    def _extract_snippet(
        doc: fitz.Document,
        *,
        page_num: int,
        rect: Optional[fitz.Rect],
        keyword: str,
        chunk: Optional[Dict[str, Any]] = None,
        match_span: Optional[Sequence[int]] = None,
        context_window: int = 160,
    ) -> str:
        if isinstance(page_num, int) and 0 <= page_num < doc.page_count:
            page = doc[page_num]
            if rect and not rect.is_empty:
                expanded = fitz.Rect(rect)
                expanded.x0 = max(expanded.x0 - 5, 0)
                expanded.x1 = expanded.x1 + 5
                expanded.y0 = max(expanded.y0 - 5, 0)
                expanded.y1 = expanded.y1 + 5
                snippet = page.get_text("text", clip=expanded).strip()
                if snippet:
                    return KeywordModeService._highlight_snippet(snippet, keyword)

        if chunk and isinstance(chunk.get("text"), str):
            text = chunk["text"].replace("\n", " ")
            start, end = None, None

            if match_span and len(match_span) == 2:
                start, end = max(match_span[0], 0), min(match_span[1], len(text))
            else:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    start, end = match.start(), match.end()

            if start is not None and end is not None:
                window_start = max(start - context_window // 2, 0)
                window_end = min(end + context_window // 2, len(text))
                snippet = text[window_start:window_end].strip()
                return KeywordModeService._highlight_snippet(snippet, keyword)

        return keyword

    @staticmethod
    def _highlight_snippet(snippet: str, keyword: str) -> str:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        return pattern.sub(lambda match: f"**{match.group(0)}**", snippet)
