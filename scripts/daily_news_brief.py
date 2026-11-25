#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily multi-region news brief delivered to Telegram in Korean.

Searches recent news per region with Tavily, asks Gemini to pick and summarize
the top items from those search results only, then sends one Telegram message.
Intended to run as a best-effort step after the daily analysis workflow.

Required env: TAVILY_API_KEYS (comma separated), GEMINI_API_KEY, GEMINI_MODEL,
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID. Optional: GEMINI_MODEL_FALLBACK,
TELEGRAM_MESSAGE_THREAD_ID.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

# (key, label, item count, search query)
REGIONS: List[Tuple[str, str, int, str]] = [
    ("kr", "🇰🇷 한국", 3, "한국 증시 경제 주요 뉴스 Korea economy KOSPI top news"),
    ("us", "🇺🇸 미국", 2, "US economy stock market Federal Reserve top news"),
    ("cn", "🇨🇳 중국", 2, "China economy stock market top news"),
    ("eu", "🇪🇺 유럽", 2, "Europe economy ECB stock market top news"),
    ("jp", "🇯🇵 일본", 2, "Japan economy Nikkei Bank of Japan top news"),
]

TAVILY_URL = "https://api.tavily.com/search"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_LIMIT = 3800
RESULTS_PER_REGION = 6
CONTENT_CHARS = 300


def search_region(query: str, keys: List[str]) -> List[Dict[str, str]]:
    """Return recent news results for one query, trying each Tavily key in turn."""
    for key in keys:
        try:
            resp = requests.post(
                TAVILY_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "query": query,
                    "topic": "news",
                    "days": 2,
                    "max_results": RESULTS_PER_REGION,
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            return [
                {
                    "title": str(r.get("title") or "").strip(),
                    "url": str(r.get("url") or "").strip(),
                    "content": str(r.get("content") or "").strip()[:CONTENT_CHARS],
                }
                for r in results
                if r.get("url") and r.get("title")
            ]
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "n/a")
            print(f"[news] tavily failed ({type(exc).__name__}, status={status})", file=sys.stderr)
    return []


def build_prompt(sources: Dict[str, List[Dict[str, str]]]) -> str:
    counts = {key: count for key, _, count, _ in REGIONS}
    return (
        "You are a news editor. From the search results below, pick the most important "
        "market- or economy-relevant news per region and write in Korean.\n"
        f"Items per region: {json.dumps(counts)}.\n"
        "Rules:\n"
        "- Use ONLY the provided results. Never invent facts, numbers or URLs.\n"
        "- Each item: headline (max 40 Korean characters) and summary (one sentence, max 80 characters).\n"
        "- url must be copied exactly from the chosen result.\n"
        "- If a region has fewer good results, return fewer items.\n"
        'Output strict JSON: {"<region>": [{"headline": "...", "summary": "...", "url": "..."}]}.\n\n'
        f"Search results:\n{json.dumps(sources, ensure_ascii=False)}"
    )


def summarize(sources: Dict[str, List[Dict[str, str]]], api_key: str, models: List[str]) -> Dict[str, Any]:
    prompt = build_prompt(sources)
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    for model in models:
        try:
            resp = requests.post(
                GEMINI_URL.format(model=model),
                headers={"x-goog-api-key": api_key},
                json=body,
                timeout=90,
            )
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return parse_json_block(text)
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "n/a")
            print(f"[news] gemini {model} failed ({type(exc).__name__}, status={status})", file=sys.stderr)
    return {}


def parse_json_block(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    data = json.loads(cleaned.strip())
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def validate_items(
    raw_items: Any, allowed_urls: set, limit: int
) -> List[Dict[str, str]]:
    """Keep only well-formed items whose url came from the search results."""
    items: List[Dict[str, str]] = []
    if not isinstance(raw_items, list):
        return items
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        headline = str(item.get("headline") or "").strip()
        summary = str(item.get("summary") or "").strip()
        url = str(item.get("url") or "").strip()
        if headline and summary and url in allowed_urls:
            items.append({"headline": headline, "summary": summary, "url": url})
        if len(items) >= limit:
            break
    return items


def format_sections(
    digest: Dict[str, Any], sources: Dict[str, List[Dict[str, str]]]
) -> List[str]:
    sections: List[str] = []
    for key, label, count, _ in REGIONS:
        allowed = {r["url"] for r in sources.get(key, [])}
        items = validate_items(digest.get(key), allowed, count)
        if not items:
            continue
        lines = [label]
        for idx, item in enumerate(items, 1):
            lines.append(f"{idx}. {item['headline']}")
            lines.append(f"   {item['summary']}")
            lines.append(f"   {item['url']}")
        sections.append("\n".join(lines))
    return sections


def chunk_message(header: str, sections: List[str], footer: str) -> List[str]:
    chunks: List[str] = []
    current = header
    for section in sections:
        candidate = f"{current}\n\n{section}"
        if len(candidate) > TELEGRAM_LIMIT and current != header:
            chunks.append(current)
            current = section
        else:
            current = candidate
    chunks.append(f"{current}\n\n{footer}")
    return chunks


def send_telegram(token: str, chat_id: str, text: str, thread_id: Optional[str]) -> bool:
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if thread_id:
        payload["message_thread_id"] = thread_id
    try:
        resp = requests.post(TELEGRAM_URL.format(token=token), json=payload, timeout=30)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", "n/a")
        print(f"[news] telegram failed ({type(exc).__name__}, status={status})", file=sys.stderr)
        return False


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    tavily_keys = [k.strip() for k in os.getenv("TAVILY_API_KEYS", "").split(",") if k.strip()]
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    models = [m.strip() for m in (os.getenv("GEMINI_MODEL", ""), os.getenv("GEMINI_MODEL_FALLBACK", "")) if m.strip()]
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    thread_id = os.getenv("TELEGRAM_MESSAGE_THREAD_ID", "").strip() or None

    missing = [
        name
        for name, value in (
            ("TAVILY_API_KEYS", tavily_keys),
            ("GEMINI_API_KEY", gemini_key),
            ("GEMINI_MODEL", models),
            ("TELEGRAM_BOT_TOKEN", token),
            ("TELEGRAM_CHAT_ID", chat_id),
        )
        if not value
    ]
    if missing:
        print(f"[news] missing env: {', '.join(missing)}", file=sys.stderr)
        return 1

    sources = {key: search_region(query, tavily_keys) for key, _, _, query in REGIONS}
    if not any(sources.values()):
        print("[news] no search results", file=sys.stderr)
        return 1

    digest = summarize(sources, gemini_key, models)
    sections = format_sections(digest, sources)
    if not sections:
        print("[news] nothing to send after validation", file=sys.stderr)
        return 1

    now = datetime.now(ZoneInfo("Asia/Seoul"))
    header = f"📰 오늘의 주요 뉴스 ({now:%m/%d %H:%M} KST)"
    footer = "※ 검색 결과를 AI가 요약한 내용으로 부정확할 수 있습니다. 원문 링크를 확인하세요."
    messages = chunk_message(header, sections, footer)

    if dry_run:
        print("\n\n----\n\n".join(messages))
        return 0
    ok = all(send_telegram(token, chat_id, text, thread_id) for text in messages)
    print(f"[news] sent {len(messages)} message(s): {'ok' if ok else 'failed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
