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

def getEmails():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    service = build('gmail', 'v1', credentials=creds)
    result = service.users().messages().list(userId='me', maxResults=50).execute()
    messages = result.get('messages', [])

    email_list = []
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = txt.get('payload', {})
        headers = payload.get('headers', [])
        subject, sender, body = "No Subject", "Unknown Sender", "No Message"
        for d in headers:
            if d['name'] == 'Subject':
                subject = d['value']
            if d['name'] == 'From':
                sender = d['value']
        body_data = "No Content"
        if 'parts' in payload:
            try:
                data = payload['parts'][0]['body']['data']
                data = data.replace("-", "+").replace("_", "/")
                decoded_data = base64.b64decode(data).decode("utf-8")
                soup = BeautifulSoup(decoded_data, "html.parser")
                body_data = soup.get_text()
            except Exception as e:
                body_data = f"Error reading content: {str(e)}"
        email_list.append({"subject": subject, "from": sender, "body": body_data})
    
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
    df_infer = pd.DataFrame([[processed_text, keyword_feature, sentiment_score]],
                            columns=['processed_text', 'keyword_feature', 'sentiment'])
    return df_infer

def classify_urgency(email_data):
    """Predict urgency using the ensemble model and map numeric output to labels."""
    df_infer = create_inference_df(email_data["subject"], email_data["body"])
    prediction = ensemble_model.predict(df_infer)[0]
    mapping = {
        3: "Non-Urgent",
        2: "Low Urgency",
        1: "Medium Urgency",
        0: "High Urgency"
    }
    return mapping.get(prediction, "Unknown")

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
