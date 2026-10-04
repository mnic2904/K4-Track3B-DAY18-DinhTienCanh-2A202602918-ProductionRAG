from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def _summary_fallback(text: str) -> str:
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+|\n+', text)
                 if part.strip()]
    return " ".join(sentences[:2]) if sentences else text


def _questions_fallback(text: str, n_questions: int) -> list[str]:
    sentences = [part.strip() for part in re.split(r'[.!?\n]+', text)
                 if len(part.strip()) > 10]
    return [f"{sentence}?" for sentence in sentences[:max(n_questions, 0)]]


def _context_fallback(text: str, document_title: str) -> str:
    prefix = f"Trích từ {document_title}.\n\n" if document_title else ""
    return f"{prefix}{text}"


def _metadata_fallback() -> dict:
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content)
    value = json.loads(content)
    if not isinstance(value, dict):
        raise TypeError("Enrichment response must be a JSON object")
    return value


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as error:  # noqa: BLE001 - fall back when API is unavailable
            print(f"  ⚠️  OpenAI summarize failed: {error}")
    return _summary_fallback(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = response.choices[0].message.content.strip().splitlines()
            return [question.strip().lstrip("0123456789.-) ")
                    for question in questions if question.strip()][:n_questions]
        except Exception as error:  # noqa: BLE001 - fall back when API is unavailable
            print(f"  ⚠️  OpenAI HyQA failed: {error}")
    return _questions_fallback(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = response.choices[0].message.content.strip()
            return f"{context}\n\n{text}" if context else text
        except Exception as error:  # noqa: BLE001 - fall back when API is unavailable
            print(f"  ⚠️  OpenAI contextual failed: {error}")
    return _context_fallback(text, document_title)


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return _parse_json(response.choices[0].message.content)
        except Exception as error:  # noqa: BLE001 - fall back when API is unavailable
            print(f"  ⚠️  OpenAI metadata failed: {error}")
    return _metadata_fallback()


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và chỉ trả về JSON hợp lệ:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            result = _parse_json(response.choices[0].message.content)
            if not all(key in result for key in ("summary", "questions", "context", "metadata")):
                raise ValueError("Enrichment response is missing required fields")
            return result
        except Exception as error:  # noqa: BLE001 - fall back when API is unavailable
            print(f"  ⚠️  Enrichment API failed: {error}")

    contextual = _context_fallback(text, source)
    context = contextual.removesuffix(text).strip()
    return {
        "summary": _summary_fallback(text),
        "questions": _questions_fallback(text, 3),
        "context": context,
        "metadata": _metadata_fallback(),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
