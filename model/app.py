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

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
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
    df_infer = create_inference_df(email_data["subject"], email_data["body"])
    prediction = ensemble_model.predict(df_infer)[0]
    
    print("\n--- Model Prediction Debug ---")
    print(f"Raw Prediction Output: {prediction}")
    
    mapping = {
        3: "Non-Urgent",
        2: "Low Urgency",
        1: "Medium Urgency",
        0: "High Urgency"
    }
    urgency_label = mapping.get(prediction, "Unknown")
    
    print(f"Mapped Urgency Label: {urgency_label}")
    
    return urgency_label


def urgency_color(urgency_label):
    """Return a color code based on urgency label."""
    color_map = {
        "High Urgency": "red",
        "Medium Urgency": "orange",
        "Low Urgency": "green",
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

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email Urgency Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .email { border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
            .subject { font-weight: bold; }
            .from { color: #555; }
            .urgency {
                font-weight: bold;
                /* We'll inject color inline to match the urgency level */
            }
        </style>
    </head>
    <body>
        <h1>Recent Emails</h1>
        {% for email in emails %}
            <div class="email">
                <div class="subject">{{ email.subject }}</div>
                <div class="from">From: {{ email.from }}</div>
                <div class="urgency" style="color: {{ email.urgency_color }};">
                    Urgency: {{ email.urgency_label }}
                </div>
                <div class="body">{{ email.body[:500] }}...</div>
            </div>
        {% endfor %}
    </body>
    </html>
    """
    return render_template_string(html_template, emails=emails)

if __name__ == '__main__':
    app.run(debug=True)
