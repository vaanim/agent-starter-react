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

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Paige, a virtual dental office assistant. 
            Your job is to:
            - Provide the office hours clearly and politely.
            - Give the office address when asked.
            - Ask for the caller's name and phone number if they want to schedule an appointment.
            - Politely confirm that a real person will call them back for scheduling.
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
    
    #this function would be moved to be used for the AppointmentAgent class.
    @function_tool()
    async def record_appointment_request(
        self,
        context: RunContext,
        name: str,
        phone: str,
        notes: Optional[str] = None
    ) -> dict[str, Any]:
        """Record a caller's name and phone number for appointment follow-up."""

        # Save to a CSV file (eventually this would go to some sort of database)
        with open("appointments.csv", "a") as f:
            f.write(f"{name},{phone},{notes or ''}\n")
        return f"Thank you {name}, your contact info has been recorded. Someone will call you as soon as possible."


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
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        stt=assemblyai.STT(),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        llm=openai.LLM(model="gpt-4o-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        tts=elevenlabs.TTS(),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
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
