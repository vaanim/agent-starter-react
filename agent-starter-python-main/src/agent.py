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
session_id = ''
N8N_URL = "https://railway.assigncorp.com/webhook/appointment-agent"

def send_to_n8n(command: str, query: str, trace_id: str | None = None):

    payload = {
        "command": command,
        "query": query
    }
    if trace_id:
        payload["trace_id"] = trace_id
    try:
        response = requests.post(
            N8N_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        response.raise_for_status()

        return response.json()
    except Exception as e:
        logger.error(f"N8N error: {e}")
        return {"status": "error", "message": "Unable to create task right now"}

class GeneralAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Paige, a virtual dental office assistant. 
            Your job is to:
            - Provide the office hours clearly and politely.
            - Give the office address when asked.
            - If they have any interests in scheduling, rescheduling, or cancelling appointments call on the 'appointment_requested' function
            - Keep responses short, polite, and professional.""",
        )
    @function_tool()
    async def get_office_hours(
        self,
        context: RunContext,
    ) -> dict[str, Any]:
        """Get the office hours of the dental office."""
        return "Our office hours are Monday to Friday, 9 AM to 5 PM."
    
    @function_tool()
    async def get_office_address(
        self, 
        context: RunContext
    ) -> dict[str, Any]:
        """Return the dental office address."""
        return "Our office is located at 123 Four Street, FiveField, California."
    
    #this function will be used to call the AppointmentAssistant class.
    @function_tool()
    async def appointment_requested(
        self,
        _context: RunContext
    ) -> str:
        """Send caller for appointment making with other agent."""
        
        logger.info("switching to the appointment assistant")
        return AppointmentAssistant(), "Of course! I'll connect you with our appointment assistant."

class AppointmentAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are Hailey, an appointment scheduling assistant for a dental office.

            When a caller wants to schedule, reschedule, or cancel an appointment (or any other request),
            collect their information ONE question at a time in this order:
            1. Ask for their full name.
            2. Ask for their phone number.
            3. Ask for their preferred date.
            4. Ask for their preferred time.

            Wait for the caller to answer each question before asking the next one.
            Do NOT ask multiple questions in the same response.

            Once you have all four pieces of information, call the create_task tool with the query formatted as:
                Name, Phone, Request, Preferred date, Preferred time

            After the task is recorded, ask the caller if there is anything else you can help them with.
            If they say no or indicate they are done, say a polite goodbye and call the end_call tool.

            Keep responses short, polite, and professional.
            """,
        )
    async def on_enter(self):
        # when the agent is added to the session, it'll initiate the conversation
        self.session.generate_reply()

    @function_tool()
    async def record_appointment_request(
        self,
        context: RunContext,
        name: str,
        phone: str,
        request_type: str,  # schedule / reschedule / cancel
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record an appointment request including type and notes."""

        request_type = request_type.lower().strip()

        if request_type not in ["schedule", "reschedule", "cancel"]:
            return "Sorry, I didn't understand the request type."

        # Save to CSV
        with open("appointments.csv", "a") as f:
            f.write(f"{name},{phone},{request_type},{notes or ''}\n")

        return (
            f"Thank you {name}. Your {request_type} request has been recorded. "
            "Someone from our office will call you as soon as possible to confirm."
        )
    @function_tool()
    async def end_call(
        self,
        context: RunContext,
    ) -> None:
        """End the call after the caller has no further needs."""
        await self.session.aclose()

    @function_tool()
    async def create_task(
        self,
        context: RunContext,
        query: str
    ) -> str:
        """
        Create a CRM task from a patient request.
        """
        result = send_to_n8n(
            command="create_task",
            query=query,
            trace_id=session_id
        )
        if isinstance(result, dict) and result.get("status") == "success":
            return "Your request has been sent to our office team. Someone will contact you shortly."

        return "I've recorded your request and our staff will follow up soon."
        
server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm



async def on_session_end(ctx: JobContext) -> None:
    report = ctx.make_session_report()
    os.makedirs("transcripts", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcripts/{ctx.room.name}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Transcript saved to {filename}")


@server.rtc_session(agent_name="my-agent", on_session_end=on_session_end)
async def my_agent(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    global session_id
    session_id = ctx.room.name
    # Set up a voice AI pipeline using OpenAI, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text: turning the user's speech into text that the LLM can understand
        stt=assemblyai.STT(),
        # Large Language Model: processing user input and generating a response
        llm=openai.LLM(model="gpt-4o-mini"),
        # Text-to-speech: turning the LLM's text into speech that the user can hear
        tts=elevenlabs.TTS(),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        preemptive_generation=True,
    )

    # Start the session
    await session.start(
        agent=GeneralAssistant(),
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
