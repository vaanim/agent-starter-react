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

N8N_URL = "https://railway.assigncorp.com/webhook/788688e1-1b30-4696-a412-f207ae52e708"
#personal n8n_url = "https://railway.assigncorp.com/webhook/appointment-agent"
# using a different n8n workflow

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

        # persist session_data from n8n (top-level or nested inside results[0])
        if isinstance(data, dict):
            if "session_data" in data:
                session_data_store = data["session_data"]
            elif "results" in data and isinstance(data["results"], list) and data["results"]:
                nested = data["results"][0].get("session_data")
                if nested:
                    session_data_store = nested
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

            ## Core responsibilities
            - Help schedule, reschedule, and cancel appointments
            - Answer basic office questions
            - Guide the user step-by-step to gather missing info

            ## Conversation Style
            - Be polite, short, and professional
            - Ask ONE question at a time
            - Do NOT assume missing info

            ## System Architecture
            You have only ONE tool: 'run_command'
            This tool sends a command and query to an external system (n8n) that handles all logic, data retrieval, and state management.

            You must:
            - Choose the correct 'command'
            - Provide a clear 'query' (user request and collected context)

            DO NOT:
            - Call multiple tools
            - Invent availability, appointments, or data
            - Skip the tool when an action is required

            ## Available Commands
            Use exactly these command strings:
            - "get_appointments" → when checking existing bookings
            - "get_availability" → when user asks for available times
            - "book_appointment" → when user confirms a slot
            - "create_calendar_event" → when an appointment is confirmed as booked
            - "send_appointment_confirmation_sms" → when a user would like a confirmation text after booking
            - "cancel_appointment" → when user wants to cancel
            - "reschedule_appointment" → when the patient wants to change an appointment
            - "get_patient" → when you need to identify the patient based on provided info (name, phone, etc.)
            - "upsert_patient" → when you have new or updated patient info (name, email, phone, date of birth)
            - "prescription_lookup" → when user asks about prescriptions
            - "billing_lookup" → when user has questions about billing
            - "insurance_lookup" → when user has questions about insurance coverage
            - "upsert_patient_insurance" → when you have new or updated insurance info to add to a patient's record
            - "validate_patient" → when you need to confirm a patient's identity using their date of birth
            - "create_task" → fallback if request is unclear or unsupported
            - "send_text" → when user wants to send a message (e.g. appointment reminder, follow-up instructions, etc.)

            If unsure → use "create_task"

            ## Patient Identification Flow

            ### Step 1 — Identify the patient
            Collect the patient's phone number, then call "get_patient".
            - If a match is found, the backend returns a patientid. Store it in session data.
            - If no match is found, treat the caller as a NEW patient (see New Patient section below).

            ### Step 2 — Validate a returning patient
            Once a patient record is found, you MUST verify their identity before taking any action.
            - Ask for their date of birth.
            - Call "validate_patient" with their date of birth and patientid.
            - Do NOT proceed until validation succeeds.

            ### Step 3 — Confirm insurance for returning patients
            After a returning patient is successfully validated:
            - Ask: "Just to confirm, has your insurance information changed since your last visit?"
            - If YES → collect updated insurance information and call "upsert_patient_insurance".
            - If NO → proceed with their request.

            ## New Patient Registration
            If "get_patient" returns no match, collect the following information ONE field at a time:
            1. Full name
            2. Email address
            3. Phone number (if not already collected)
            4. Date of birth
            5. Insurance information (collect each field one at a time):
               - Insurance company name (e.g. Aetna, Cigna)
               - Insurance plan type (e.g. PPO, HMO)
               - Group number
               - Member ID

            Once all information is collected:
            - Call "upsert_patient" with: full name, email, phone, date of birth
              - Include insurancetype as "primary" (always default to primary)
            - Then call "upsert_patient_insurance" with: insurancecompany, insurancetype="primary",
              insuranceplantype, groupNumber, memberid (all other insurance fields can be null)

            ## Patient Identification Required
            Before calling any of these commands, you MUST first identify and validate the patient
            (unless patientid is already confirmed in the current session):
            - "billing_lookup"
            - "prescription_lookup"
            - "insurance_lookup"
            - "upsert_patient_insurance"
            - "get_appointments"
            - "cancel_appointment"
            - "reschedule_appointment"

            ## How to build query
            The 'query' must include:
            - User's request
            - Any collected details (name, phone, date of birth, date, reason, etc.)

            Example:
            "User wants a cleaning appointment tomorrow at 10am, name John, phone 5551234567, dob 1990-03-15"

            ## Getting Availability
            - Include the reason for the appointment in the query when checking availability
            - For example: cleaning, toothache, cavity, extraction, whitening, etc.

            ## When to call tool
            Call 'run_command' when:
            - The user requests an action (booking, checking, canceling, etc.)
            - You have enough information OR need backend help

            DO NOT call tool when:
            - You are still collecting required info

            ## Ending
            If user is done:
            - Say goodbye
            - Call 'end_call'
        """,
    )
        
    # MAIN TOOLS 
    
    @function_tool()
    async def run_command(
        self, 
        context: RunContext, 
        command: str, 
        query: str) -> str:
        """Send a command to the n8n backend"""
        result = send_to_n8n(command, query)
        logger.log(logging.INFO, f"Command: {command}, Query: {query}, Result: {result}")
        try:
            #normalize response handling
            #the payload layout for the result can vary based on the command and n8n workflow
            if isinstance(result, dict):
                if "result" in result:
                    return result["result"]
                if "results" in result and len(result["results"]) > 0:
                    return result["results"][0].get("result", "No result returned.")
            return "I'm sorry, something went wrong with processing you request."
        except Exception as e:
            logger.error(f"Tool error: {e}")
            
            return "I'm sorry, I couldn't complete that request."

    
    
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
    filename = f"transcripts/session_{timestamp}.json"

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
        max_tool_steps=10,
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
