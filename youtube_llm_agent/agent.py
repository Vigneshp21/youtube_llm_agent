from google.adk.agents import Agent
from .tools import process_youtube_video

root_agent = Agent(
    name="youtube_llm_transcriber",
    description=(
        "Agent that takes a YouTube URL and returns a transcript and summary."
    ),
    model="gemini-2.5-flash",
    tools=[process_youtube_video],
    instruction=(
        "You are a helpful agent that specializes in YouTube analysis. "
        "When the user provides a YouTube URL, ALWAYS call the function tool "
        "`process_youtube_video` with the URL. Return the tool's JSON output to the user."
        "If the user writes anything else, ask them for a valid YouTube URL."
    ),
)

