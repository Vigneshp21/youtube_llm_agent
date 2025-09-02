import argparse
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from youtube_llm_agent.tools import process_youtube_video


def main():
    parser = argparse.ArgumentParser(description="YouTube LLM Transcriber (ADK)")
    parser.add_argument("--url", required=True, help="Public YouTube video URL")
    args = parser.parse_args()

    result = asyncio.run(process_youtube_video(args.url))
    transcript_head = (result.get("transcript") or "")[:800]
    summary = result.get("summary") or ""
    error = result.get("error")

    print("\n=== Summary ===\n")
    print(summary if summary else "(no summary)")
    print("\n=== Transcript (first 800 chars) ===\n")
    print(transcript_head if transcript_head else "(no transcript)")
    if error:
        print("\n[Error]\n" + error)


if __name__ == "__main__":
    main()
