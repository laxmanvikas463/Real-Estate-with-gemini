import json
import random
import requests
import torch
import numpy as np
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem.porter import PorterStemmer

# NLTK downloads
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Flask app setup
app = Flask(__name__)
CORS(app)

# WhatsApp Cloud API credentials
VERIFY_TOKEN = "VspD-4Db!4u]"
WHATSAPP_TOKEN = "EAAJZAa4DFt8wBO5fuanHSWsrog51y32RAu5MirMyl9naJrRlW5vJTKpIZCUHOsVurdiwQMUOxD2nI3Iy8xXEayr95JIAg9KZCq8auB7xfkNfJ2KbDZB5KTGeBnIWfm5lAFh0ZARa1VtObJdbob50czOFFuqSRhA5fXRZAH0zGnGdzVZAoUZAtUXSW7JZAuhvYqkHcwDFv9eFvjNH1ZACzkD1wZBm6ILV4HA69sH8xSPBTuwmDgiSZCAZD"
PHONE_NUMBER_ID = "582940438245620"

# NLP tools
stemmer = PorterStemmer()
def tokenize(sentence): return nltk.word_tokenize(sentence)
def stem(word): return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Load trained model
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('real intents.json', 'r') as f:
    intents = json.load(f)

data = torch.load("data.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(data["model_state"])
model.eval()

# Chatbot logic
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    prob = torch.softmax(output, dim=1)[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])
    return "I do not understand..."

# WhatsApp message sender
def send_whatsapp_message(phone_number_id, to_number, message):
    url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "text": {
            "body": message
        }
    }
    res = requests.post(url, headers=headers, json=payload)
    print("Message sent:", res.status_code, res.text)

# Basic test route
@app.route('/')
def index():
    return jsonify({"message": "Flask on Cloud Run is live!"})

# Chat test endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    response = get_response(message)
    return jsonify({"response": response})

# Webhook for WhatsApp
# @app.route('/webhook', methods=['GET', 'POST'])
# def webhook():
#     if request.method == 'GET':
#         mode = request.args.get("hub.mode")
#         token = request.args.get("hub.verify_token")
#         challenge = request.args.get("hub.challenge")
#         if mode == "subscribe" and token == VERIFY_TOKEN:
#             return challenge, 200
#         else:
#             return "Verification failed", 403

#     if request.method == 'POST':
#         data = request.get_json()
#         for entry in data.get("entry", []):
#             for change in entry.get("changes", []):
#                 value = change.get("value", {})
#                 messages = value.get("messages", [])
#                 if messages:
#                     msg = messages[0]
#                     from_number = msg["from"]
#                     text = msg["text"]["body"]
#                     reply = get_response(text)
#                     send_whatsapp_reply(from_number, reply)
#         return "OK", 200
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        # Facebook webhook verification
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')

        if mode == 'subscribe' and token == VERIFY_TOKEN:
            print("WEBHOOK VERIFIED")
            return challenge, 200
        else:
            return 'Verification token mismatch', 403

    elif request.method == 'POST':
        data = request.get_json()
        print("Received webhook:", data)  # useful for debugging

        if data.get("object") == "whatsapp_business_account":
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])
                    if messages:
                        for message in messages:
                            phone_number_id = value["metadata"]["phone_number_id"]
                            from_number = message["from"]  # sender's phone number
                            user_msg = message["text"]["body"]  # message sent

                            # Generate response from your chatbot
                            bot_response = get_response(user_msg)

                            # Send response back using WhatsApp API
                            send_whatsapp_message(phone_number_id, from_number, bot_response)

        return 'EVENT_RECEIVED', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
