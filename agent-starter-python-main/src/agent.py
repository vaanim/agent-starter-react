import logging
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

logger = logging.getLogger("agent")

load_dotenv(".env.local")


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

            Your job is to:
            1. Ask for the caller's first and last name.
            2. Ask for a good phone number to call back.
            3. Ask what they want to do: schedule, reschedule, or cancel.
            4. If schedule → ask what dates and times work best.
            5. If reschedule → ask for their current appointment date/time AND preferred new times.
            6. If cancel → ask for the appointment date/time. if they do not know, write it in as notes and let them know someone will call them back.
            7. Collect any extra notes.
            8. Confirm politely that a real person will call them back to confirm their request.

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
server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

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
