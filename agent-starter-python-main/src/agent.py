import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    room_io,
)
from livekit.plugins import assemblyai, elevenlabs, openai
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from opentelemetry import context

import requests

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# GLOBALS #
session_id = ''
session_data_store = {}

N8N_URL = "https://railway.assigncorp.com/webhook-test/788688e1-1b30-4696-a412-f207ae52e708"
#personal n8n_url = "https://railway.assigncorp.com/webhook/appointment-agent"
# using a different n8n workflow now!

# HELPER METHODS #

# extract conversation history from the session report to send to n8n and save locally as a transcript
def extract_conversation(report_dict):
    messages = []
    items = report_dict.get("chat_history", {}).get("items", [])

    for item in items:
        if item.get("type") == "message":
            role = item.get("role")
            content = " ".join(item.get("content", []))
            timestamp = item.get("created_at")

            readable_time = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

            messages.append({
                "time": readable_time,
                "role": role,
                "message": content
            })

    return messages

#currently not in use
def build_call_metadata(conversation):
    full_text = " ".join([msg["message"] for msg in conversation]).lower()

    return {
        "intent": "appointment_request" if "appointment" in full_text else "unknown",
        "has_phone": any(char.isdigit() for char in full_text),
        "message_count": len(conversation)
    }

def send_to_n8n(
    command: str,
    query: str | None = None,
):
    global session_data_store

    payload = {
        "command": command,
        "query": query or "",
        "sessionid": session_id,
        "session_data": session_data_store or {}
    }

    try:
        response = requests.post(
            N8N_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=20
        )

        response.raise_for_status()
        data = response.json()

        # persist session_data from n8n
        if isinstance(data, dict) and "session_data" in data:
            session_data_store = data["session_data"]
        print(data)
        return data

    except Exception as e:
        logger.error(f"N8N error: {e}")
        return {"status": "error", "message": "Request failed"}

# END HELPER METHODS #

#SINGLE AGENT IMPLEMENTATION

class DentalAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are Paige, a front desk assistant for a dental office.

            ## Responsibilities
            - Help schedule, reschedule, and cancel appointments
            - Answer basic office questions
            - Guide the user step-by-step to gather missing info

            ## Behavior
            - Be polite, short, and professional
            - Ask ONE question at a time
            - Do NOT assume missing info

            ## Tool Usage
            - get_availability → when user asks for times
            - book_appointment → when user confirms a slot
            - create_task → fallback if unclear
            - cancel_appointment / reschedule_appointment → when requested

            Always rely on tool responses. Never make up availability.

            ## Getting Availability
            - make sure to include the reason in for the appointment in the query when checking availability
            - for example: cleaning, toothache, cavity, extraction, whitening, etc.

            ## Ending
            If user is done → say goodbye and call end_call
""",
    )
        
       # MAIN TOOLS 
    
    @function_tool()
    async def get_availability(self, context: RunContext, query: str) -> str:
        """Get the office hours of the dental office."""
        result = send_to_n8n("get_availability", query)
        try:
            return result["results"][0]["result"]
        except Exception as e:
            logger.error(f"Error retrieving availability: {e}")
            return "I'm sorry, I couldn't retrieve availability right now."
    
    @function_tool()
    async def book_appointment(self, context: RunContext, query: str) -> str:
        result = send_to_n8n("book_appointment", query)
        return result.get("result", "I couldn't complete the booking.")

    @function_tool()
    async def cancel_appointment(self, context: RunContext, query: str) -> str:
        result = send_to_n8n("cancel_appointment", query)
        return result.get("result", "Unable to cancel appointment.")

    @function_tool()
    async def reschedule_appointment(self, context: RunContext, query: str) -> str:
        result = send_to_n8n("reschedule_appointment", query)
        return result.get("result", "Unable to reschedule appointment.")

    @function_tool()
    async def create_task(self, context: RunContext, query: str) -> str:
        result = send_to_n8n("create_task", query)
        return result.get("result", "I've recorded your request. Our team will follow up.")

    @function_tool()
    async def end_call(self, context: RunContext) -> None:
        await self.session.aclose()

    @function_tool()
    async def get_office_address(
        self, 
        context: RunContext
    ) -> dict[str, Any]:
        """Return the dental office address."""
        return "Our office is located at 123 Four Street, FiveField, California."
    
# SERVER SETUP     
server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

async def on_session_end(ctx: JobContext) -> None:
    report = ctx.make_session_report()
    clean_conversation = extract_conversation(report.to_dict())

    os.makedirs("transcripts", exist_ok=True)
    timestamp = datetime.now().strftime("%m_%d_%Y_%H%M")
    filename = f"transcripts/{ctx.room.name}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(clean_conversation, f, indent=2)

    logger.info(f"Transcript saved to {filename}")

    send_to_n8n(
        command="end-of-call-report",
        query="Call ended"
    )

@server.rtc_session(agent_name="my-agent", on_session_end=on_session_end)

async def my_agent(ctx: JobContext):
    global session_id, session_data_store

    session_id = ctx.room.name
    session_data_store = {}  # reset per call

    session = AgentSession(
        stt=assemblyai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Start the session
    await session.start(
        agent=DentalAssistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)
