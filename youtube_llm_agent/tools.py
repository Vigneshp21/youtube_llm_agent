from __future__ import annotations

import os
import re
from typing import Dict, Any

from google.adk.models import Gemini

_YT_REGEX = re.compile(r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+", re.I)

def _get_model_id() -> str:
    return os.getenv("ADK_MODEL", "gemini-2.5-flash")

def _extract_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    for attr in ("text", "output_text", "content", "response"):
        if hasattr(resp, attr) and isinstance(getattr(resp, attr), str):
            return getattr(resp, attr)
    try:
        candidates = getattr(resp, "candidates", None)
        if candidates:
            cand = candidates[0]
            content = getattr(cand, "content", None)
            if content and getattr(content, "parts", None):
                for p in content.parts:
                    t = getattr(p, "text", None)
                    if isinstance(t, str) and t.strip():
                        return t
    except Exception:
        pass
    return str(resp)

async def _consume_async_gen(gen) -> str:
    texts = []
    async for chunk in gen:
        t = _extract_text(chunk)
        if t:
            texts.append(t)
    return "\n".join(texts).strip()

async def process_youtube_video(url: str) -> Dict[str, str]:
    if not isinstance(url, str) or not _YT_REGEX.match(url.strip()):
        return {
            "transcript": "",
            "summary": "",
            "error": "Please provide a valid, public YouTube video URL."
        }

    model = Gemini(model=_get_model_id())

    transcript_prompt = (
        "You will be provided a public YouTube URL.\n"
        "Task: Listen to the video's audio and produce the *full transcript* "
        "faithfully, in the video's original language. Do not summarize, do not "
        "paraphrase. Include utterances in order; omit filler like long pauses. "
        "Return only the transcript text.\n\n"
        f"URL: {url}"
    )
    try:
        transcript_gen = model.generate_content_async(transcript_prompt)
        transcript_text = await _consume_async_gen(transcript_gen)
    except Exception as e:
        return {
            "transcript": "",
            "summary": "",
            "error": f"Gemini failed to generate transcript: {e}",
        }

    if not transcript_text:
        return {
            "transcript": "",
            "summary": "",
            "error": "Empty transcript received from the model. The video may be unavailable or restricted.",
        }

    summary_prompt = (
        "Summarize the following transcript of a YouTube video. "
        "Keep it concise (5-8 sentences), cover the main points, and preserve "
        "the original language unless the transcript mixes languages. "
        "If the transcript is excessively long, focus on the most salient sections.\n\n"
        f"Transcript:\n{transcript_text}"
    )
    try:
        summary_gen = model.generate_content_async(summary_prompt)
        summary_text = await _consume_async_gen(summary_gen)
    except Exception as e:
        return {
            "transcript": transcript_text,
            "summary": "",
            "error": f"Gemini failed to summarize: {e}",
        }

    return {
        "transcript": transcript_text,
        "summary": summary_text,
    }
