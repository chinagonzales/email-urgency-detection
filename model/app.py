from flask import Flask, render_template_string
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os.path
import base64
from bs4 import BeautifulSoup
import joblib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ensemble import EnhancedMNB

app = Flask(__name__)

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
TOKEN_FILE = "token.pickle"
CREDENTIALS_FILE = "credentials.json"

def getEmails():
    creds = None

    # Load existing credentials if available
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    # If credentials are invalid, refresh or reauthenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())  
                print("Token refreshed successfully.")
            except RefreshError:
                print("Token refresh failed. Deleting token and re-authenticating.")
                os.remove(TOKEN_FILE)  
                creds = None  

        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
                print("New authentication successful.")
            except Exception as e:
                print(f"Error during authentication: {e}")
                return []  

        # Save valid credentials for future use
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    # Connect to Gmail API
    try:
        service = build('gmail', 'v1', credentials=creds)
        result = service.users().messages().list(userId='me', maxResults=50).execute()
        messages = result.get('messages', [])
    except Exception as e:
        print(f"Error connecting to Gmail API: {e}")
        return []

    email_list = []
    for msg in messages:
        try:
            txt = service.users().messages().get(userId='me', id=msg['id']).execute()
            payload = txt.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = sender = body_data = "Unknown"

            # Extract subject and sender
            for d in headers:
                if d['name'] == 'Subject':
                    subject = d['value']
                if d['name'] == 'From':
                    sender = d['value']

            # Extract email body
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain': 
                        data = part['body']['data']
                        break
                    elif part['mimeType'] == 'text/html':
                        data = part['body']['data']

                if data:
                    data = data.replace("-", "+").replace("_", "/")
                    decoded_data = base64.b64decode(data).decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(decoded_data, "html.parser")
                    body_data = soup.get_text()
            else:
                body_data = "No Content"

            email_list.append({"subject": subject, "from": sender, "body": body_data})

        except Exception as e:
            print(f"Error reading email: {e}")
            continue  

    return email_list

# Load the trained ensemble model
ensemble_model = joblib.load("ensemble_model.pkl")

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()
keywords = ['urgent', 'critical', 'asap', 'immediate', 'important', 'immediately', 
            'as soon as possible', 'please reply', 'need response', 'emergency', 'high priority']

def create_inference_df(subject, body):
    text = subject + " " + body
    processed_text = ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)])
    keyword_feature = sum(1 for word in processed_text.split() if word in keywords)
    sentiment_score = (analyzer.polarity_scores(processed_text)['compound'] + 1) / 2

    print("\n--- Feature Extraction Debug ---")
    print(f"Subject: {subject}")
    print(f"Body: {body[:100]}...")  # Print first 100 characters
    print(f"Processed Text: {processed_text[:100]}...")
    print(f"Keyword Feature: {keyword_feature}")
    print(f"Sentiment Score: {sentiment_score}")
    
    df_infer = pd.DataFrame([[processed_text, keyword_feature, sentiment_score]],
                            columns=['processed_text', 'keyword_feature', 'sentiment'])
    return df_infer


def classify_urgency(email_data):
    """Predict urgency using the ensemble model and map numeric output to grouped labels."""
    df_infer = create_inference_df(email_data["subject"], email_data["body"])
    prediction = ensemble_model.predict(df_infer)[0]
    
    # group urgency levels
    if prediction in [3, 2]:
        return "Non-Urgent"
    elif prediction in [1, 0]:
        return "Urgent"
    else:
        return "Unknown"


def urgency_color(urgency_label):
    """Return a color code based on the grouped urgency label."""
    color_map = {
        "Urgent": "red",
        "Non-Urgent": "blue",
        "Unknown": "gray"
    }
    return color_map.get(urgency_label, "black")


@app.route('/')
def index():
    emails = getEmails()
    for email_item in emails:
        label = classify_urgency(email_item)
        email_item["urgency_label"] = label
        email_item["urgency_color"] = urgency_color(label)

# HTML PART/ UI
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Urgency Classifier</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            function filterEmails(urgency) {
                document.querySelectorAll('.email-card').forEach(card => {
                    if (urgency === 'All' || card.dataset.urgency === urgency) {
                        card.style.display = 'block';
                    } else {
                        card.style.display = 'none';
                    }
                });
            }

            function openModal(subject, sender, body) {
                document.getElementById("modal-subject").textContent = subject;
                document.getElementById("modal-sender").textContent = sender;
                document.getElementById("modal-body").textContent = body;
                document.getElementById("email-modal").classList.remove("hidden");
            }

            function closeModal() {
                document.getElementById("email-modal").classList.add("hidden");
            }

            function sendReply() {
                const replyText = document.getElementById("reply-text").value;
                if (replyText.trim() === "") {
                    alert("Reply cannot be empty.");
                    return;
                }
                alert("Reply Sent: " + replyText);
                document.getElementById("reply-text").value = "";
            }
        </script>
    </head>
    <body class="bg-gray-100 p-6">
        <div class="max-w-4xl mx-auto bg-white shadow-lg rounded-lg p-6">
            <div class="text-center mb-6">
                <h1 class="text-3xl font-bold text-gray-900 flex justify-center items-center gap-2">
                    <span class="material-symbols-outlined text-blue-500">support_agent</span>
                    Customer Support Inbox
                </h1>
                <p class="text-gray-500 text-sm">Manage and respond to customer inquiries</p>
            </div>
            <div class="mb-4">
                <label class="font-medium">Filter by Urgency:</label>
                <select class="border p-2 rounded" onchange="filterEmails(this.value)">
                    <option value="All">All</option>
                    <option value="Urgent">Urgent</option>
                    <option value="Non-Urgent">Non-Urgent</option>
                </select>
            </div>        
            {% for email in emails %}
                <div class="email-card border border-gray-200 p-4 mb-2 rounded-xl shadow-md bg-white cursor-pointer 
                    hover:shadow-lg hover:bg-gray-100 transition duration-300"
                    data-urgency="{{ email.urgency_label }}"
                    data-urgency-color="{{ email.urgency_color }}"
                    onclick="openModal('{{ email.subject }}', '{{ email.from }}', `{{ email.body | escape }}`)"
                    style="--urgency-border: {{ email.urgency_color }};">

                    <div class="flex flex-col space-y-1 w-full">
                        <!-- subject -->
                        <div class="font-semibold text-lg text-gray-900 truncate">{{ email.subject }}</div>

                        <!-- sender -->
                        <div class="text-gray-500 text-sm">From: {{ email.from }}</div>

                        <!-- preview of the email -->
                        <div class="text-gray-600 text-sm mt-1 line-clamp-2">
                            {{ " ".join(email.body.split()[:80]) }}{% if email.body.split()|length > 80 %}...{% endif %}
                        </div>
                    </div>

                    <!-- urgency label -->
                    <span class="px-3 py-1 text-xs font-semibold rounded-full shadow-sm"
                        style="background-color: {{ email.urgency_color }}; color: white;">
                        {{ email.urgency_label }}
                    </span>
                </div>
            {% endfor %}
        </div>

        <!-- modal/popup -->
        <div id="email-modal" class="hidden fixed inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center">
            <div class="bg-white p-6 rounded-lg shadow-lg max-w-lg w-full relative">
                <!-- close button -->
                <button onclick="closeModal()" 
                    class="absolute top-3 right-3 text-gray-500 hover:text-gray-700">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>

                <h2 id="modal-subject" class="text-xl font-bold"></h2>
                <p id="modal-sender" class="text-gray-600 text-sm mb-4"></p>
                <p id="modal-body" class="text-gray-700"></p>

                <!-- Reply Box -->
                <div class="border rounded-lg mt-4 shadow-sm">
                    <textarea id="reply-text" class="w-full p-3 border-none focus:outline-none resize-none" placeholder="Type your reply..." rows="4"></textarea>

                    <!-- toolbar -->
                    <!-- google icons -->
                    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" />

                    <div class="flex items-center justify-between p-2 border-t bg-gray-100">
                        <div class="flex space-x-3 text-gray-600">
                            <!-- Formatting Options -->
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Formatting Options">
                                <span class="material-symbols-outlined">format_color_text</span>
                            </button>
                            
                            <!-- Attach Files -->
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Attach Files">
                                <span class="material-symbols-outlined">attach_file</span>
                            </button>
                            
                            <!-- Insert Link -->
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Insert Link">
                                <span class="material-symbols-outlined">link</span>
                            </button>
                            
                            <!-- Insert Emoji -->
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Insert Emoji">
                                <span class="material-symbols-outlined">mood</span>
                            </button>
                            
                            <!-- Insert Files Using Drive -->
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Insert Files Using Drive">
                                <span class="material-symbols-outlined">drive_export</span>
                            </button>
                            
                            <!-- Insert Photo -->
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Insert Photo">
                                <span class="material-symbols-outlined">image</span> <!-- Fixed "imagesmode" typo -->
                            </button>
                            
                            <!-- Toggle Confidential Mode -->
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Toggle Confidential Mode">
                                <span class="material-symbols-outlined">lock_clock</span>
                            </button>
                            
                            <!-- Insert Signature -->
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Insert Signature">
                                <span class="material-symbols-outlined">draw</span>
                            </button>
                            
                            <!-- More Options -->
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="More Options">
                                <span class="material-symbols-outlined">more_vert</span>
                            </button>
                        </div>

                        <!-- Send Button -->
                        <button class="bg-blue-500 text-white px-4 py-2 rounded-lg shadow-md hover:bg-blue-600 focus:outline-none">
                            Send
                        </button>
                    </div>

                </div>
            </div>
        </div>
    </body>
    <style>
    /* match the border according to the urgency color */
        .email-card {
            border-color: var(--urgency-border, #e5e7eb); /* Default to gray */
            transition: border-color 0.3s ease-in-out;
        }

        .email-card:hover {
            border-color: var(--urgency-border);
        }
        </style>

    </html>

    """
    return render_template_string(html_template, emails=emails)

if __name__ == '__main__':
    app.run(debug=True)
