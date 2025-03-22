import os
import logging
import uuid
from openai import OpenAI
import json
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


logger = logging.getLogger(__name__)
client  = OpenAI()
path_to_service_json = os.path.join(os.path.dirname(__file__), 'genai-genesis-301f1-382e6569b799.json')
cred = credentials.Certificate(path_to_service_json)
firebase_admin.initialize_app(cred)
db = firestore.client()
def push_to_firebase(response):
    try:
        response["datetime"] = datetime.datetime.now().isoformat()
        response["status"] = "pending"
        response["ticket_id"] = "TICKET" + uuid.uuid4().hex

        #save to firebase
        doc_ref = db.collection('tickets').document(response['ticket_id'])
        print(response)
        doc_ref.set(response)
    except Exception as e:
        logger.error(e)


def classify_and_create_ticket(transcripts):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"based on provided transcript {str(transcripts)}, which is a communication between a victim and emergency services, produce the response in json format, to best of your knowledge with following keys 'name', 'priority', 'summary', 'services needed':[ambulance, firebrigade], 'life threatening', 'ticket_type' (one of the fire, earthquake, flood, hurricane, landslide, disease), 'smoke visibility', 'fire visibility', 'breathing issue', 'location', 'help for whom':[yourself, someone else], just give the JSON, I dont' need explanations or assumptions"
                }
            ]
        )
        response = completion.choices[0].message.content
        response = response.replace("```json", "")
        response = response.replace("```", "")
        response = response.strip()
        push_to_firebase(json.loads(response))
    except Exception as e:
        logger.error(e)
        response = {
            "None": None
        }


