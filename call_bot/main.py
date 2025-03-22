import os
import json
import base64
import asyncio
from datetime import datetime

import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
import logging

from call_bot.tools import classify_and_create_ticket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()
# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # requires OpenAI Realtime API Access
PORT = int(os.getenv('PORT', 5050))


SYSTEM_MESSAGE = (
    '''
    You are the Emergency Relief Bot. Your job is to gather critical details quickly while keeping the user calm. Follow these steps exactly, keeping all responses short, clear, and reassuring:

    Ask the user what the emergency is.
    
    Confirm their response briefly, then ask for the location and confirm that as well properly.
    
    If medical, ask about the person's condition. If fire, ask about people in danger. If crime, ask about the suspect.
    
    Confirm each response before moving on. Escalate immediately if severe, (otherwise take in information properly)  by saying Escalating Now and hangup.
    
    Once help is on the way, reassure the user and end the conversation unless they need more assistance.
    
    Maintain a composed, natural tone. Keep all responses efficient. Say goodbye once all information is collected.
  '''
)
# "Hello, I am Emergency Relief bot, and I am here to assist you in this emergency. Please stay calm, and I will gather the necessary information to ensure you get the help you need as quickly as possible. First, can you tell me your name? Thank you, [Caller’s Name]. Just to confirm, your name is [Caller’s Name], correct? Now, can you describe the nature of the emergency? I want to make sure I fully understand the situation so I can assist you effectively. Where is the emergency located? Please provide the address or as many details as possible, and I will repeat it back to confirm accuracy. Are you currently safe? Are you alone, or is someone with you? Your safety is my priority. Do you need immediate help? Is this a life-threatening situation? Can you see smoke or fire? Are you experiencing difficulty breathing? Who needs help—you or someone else? I will confirm your details as we go to ensure everything is accurate. Please know that help is already on the way. In the meantime, here are a few important safety tips: (Provide relevant guidance based on the emergency, such as staying low in smoke, avoiding unnecessary movement if injured, or remaining in a safe place.) Stay calm, [Caller’s Name]. You are not alone, and help is coming. If you need anything else, I am here for you. Stay safe, and I will now end this call unless you need further assistance. MAKE SURE YOUR QUESTIONS ARE SMALL, WITHOUT MUCH THANKS. DO REPEAT EACH DETAIL USER, IN CONSISE FASHION TELLS YOU TO CONFIRM BEFORE ASKING NEXT QUESTION"
# Use your knowledge what actual cities and states exist in the US & Canada to make sure you properly transcibe the correct city and state from the caller. Use following template - In a wildfire situation, a fire department marshal would ask the caller several key questions to assess the situation. First, they would ask for the exact location, whether the caller is in immediate danger, their name and contact number, if others are with them (including children, elderly, or disabled individuals), and whether they have a safe way out or are trapped. Next, they would inquire about the fire and smoke, including whether flames or just smoke are visible, the color of the smoke, how fast the fire is spreading, what is burning, and if there are any explosions or strange sounds. Environmental and weather conditions are also assessed, such as the direction of the wind, if there is flying debris or embers, if nearby roads are blocked, and if there are power outages or downed power lines. The marshal would also ask about evacuation and shelter status, including whether the caller has been told to evacuate, knows the nearest evacuation route, is in a safe place, has transportation, and if they are sheltering in place with enough water, food, and protective gear. Additional risks and hazards would be discussed, including the presence of hazardous materials, trapped or injured people, and pets or animals in danger. Finally, the marshal would give instructions to stay calm, prepare to evacuate or shelter if safe, cover the nose and mouth with a damp cloth to avoid smoke inhalation, and not return to the area until authorities declare it safe.
VOICE = 'alloy'
LOG_EVENT_TYPES = [
  'response.content.done', 'rate_limits.updated', 'response.done',
  'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
  'input_audio_buffer.speech_started', 'response.create', 'session.created', 'response.'
]
SHOW_TIMING_MATH = False
app = FastAPI()
if not OPENAI_API_KEY:
  raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')


@app.get("/", response_class=HTMLResponse)
async def index_page():
    return "<html><body><h1>Twilio Media Stream Server is running!</h1></body></html>"
@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    logger.info("Received incoming call request from: %s", request.client.host)
    response = VoiceResponse()
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    logger.info("Successfully created the TwiML response")
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()
    global transcripts
    transcripts = []

    # Initialize conversation transcript

    async with websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview',
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
    ) as openai_ws:
        await send_session_update(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        current_user_input = ""  # Track current user input

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }

                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
                    elif data['event'] == 'stop':
                        print("Stream has stopped.")
                        save_transcript(transcripts)

            except WebSocketDisconnect:
                print("Client disconnected.")

        async def send_to_twilio(transcripts):
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio, current_user_input
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    # Record user's transcribed speech
                    if response.get('type') == 'response.done' and len(response.get('response').get('output'))==1:
                        responsex = response.get('response')
                        transcript = responsex.get('output')[0].get('content')[0].get('transcript')

                        if transcript:  # Ensure transcript exists before appending
                            transcripts.append({'role': 'AI', 'text': transcript})
                            print(f"AI: {transcript}")

                            if "goodbye" in transcript.lower() or "end this call" in transcript.lower() or "escalating now" in transcript.lower():
                                if "escalating now" in transcript.lower():
                                    transcripts.append({'role': 'AI', 'text': "Escalated case"})
                                save_transcript(transcripts)
                                transcripts = []  # Clear transcripts after saving

                    # Record AI's response
                    if response.get('type') == 'response.done' and response.get('response').get('output'):
                        responsex = response.get('response')
                        # print(f"Response: {responsex}")
                        transcript = responsex.get('output')[0].get('content')[0].get('transcript')
                        transcripts.append(json.dumps(transcript))
                        # print(f"Transcript: {transcript}")
                        if "goodbye" in transcript.lower():
                            save_transcript(transcript)
                            transcripts = []



                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)

                        if response_start_timestamp_twilio is None:
                            response_start_timestamp_twilio = latest_media_timestamp
                            if SHOW_TIMING_MATH:
                                print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                        # Update last_assistant_item safely
                        if response.get('item_id'):
                            last_assistant_item = response['item_id']

                        await send_mark(websocket, stream_sid)

                    # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Speech started detected.")
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(
                        f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio(transcripts))

def save_transcript(transcripts):
    """
    Save the transcript of the call to a database.

    :param transcript: A list of dictionaries containing the transcript of the call
    :return: True if the transcript was saved successfully, False otherwise
    """
    try:
        # logger.info(f"Saving call transcript: {transcripts}")
        # with open("transcripts.txt", 'w') as transcript_file:
        #     for transcript in transcripts:
        #         transcript_file.write(json.dumps(transcript))
        print(transcripts)
        classify_and_create_ticket(transcripts)
        # Save the transcript to a database
    except Exception as e:
        logger.error(f"Error saving call transcript: {str(e)}")



async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! I am an Emergency Relief Bot. How can I help you today?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def send_session_update(openai_ws):
    """Send session update to OpenAI WebSocket."""

    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "server_vad"
            },
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    await send_initial_conversation_item(openai_ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)