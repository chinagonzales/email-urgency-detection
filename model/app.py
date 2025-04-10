from flask import Flask, render_template_string, jsonify, Response
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
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
from ensemble import EnhancedMNB, LogisticRegressionScratch, Ensemble
import threading
import time
import queue
import json

app = Flask(__name__)

# Gmail API scope for reading and modifying emails
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
TOKEN_FILE = "token.pickle"
CREDENTIALS_FILE = "credentials.json"

# Email tracking variables
last_email_id = None
email_cache = []

# Queue system for real-time updates
new_email_queue = queue.Queue()
waiting_clients = []

def getEmails(check_new_only=False):
    global last_email_id, email_cache
    
    creds = None

    # Load saved credentials if available
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    # Handle expired or invalid credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())  
                print("Token refreshed successfully.")
            except RefreshError:
                print("Token refresh failed. Starting fresh authentication.")
                os.remove(TOKEN_FILE)  
                creds = None  

        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=8080)
                print("New authentication successful.")
            except Exception as e:
                print(f"Authentication error: {e}")
                return []  

        # Save valid credentials for future use
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    # Fetch emails from Gmail API
    try:
        service = build('gmail', 'v1', credentials=creds)
        
        # Query setup for email fetching
        query = None
        if check_new_only and last_email_id:
            query = None
            
        result = service.users().messages().list(userId='me', maxResults=50, q=query).execute()
        messages = result.get('messages', [])
        
        # Update reference point for new emails
        if messages and not check_new_only:
            last_email_id = messages[0]['id']
            
    except Exception as e:
        print(f"Gmail API connection error: {e}")
        return []

    # Filter for only new emails when checking updates
    if check_new_only and last_email_id:
        known_ids = [email.get('id') for email in email_cache if 'id' in email]
        messages = [msg for msg in messages if msg['id'] not in known_ids]
        
        if not messages:
            return []

    # Process each email message
    email_list = []
    for msg in messages:
        try:
            txt = service.users().messages().get(userId='me', id=msg['id']).execute()
            payload = txt.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = sender = body_data = "Unknown"

            # Extract email metadata
            for d in headers:
                if d['name'] == 'Subject':
                    subject = d['value']
                if d['name'] == 'From':
                    sender = d['value']

            # Extract email content
            data = None
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain': 
                        data = part['body'].get('data')
                        break
                    elif part['mimeType'] == 'text/html':
                        data = part['body'].get('data')
            elif 'body' in payload and 'data' in payload['body']:
                data = payload['body']['data']

            if data:
                data = data.replace("-", "+").replace("_", "/")
                decoded_data = base64.b64decode(data).decode("utf-8", errors="ignore")
                soup = BeautifulSoup(decoded_data, "html.parser")
                body_data = soup.get_text()
            else:
                body_data = "No Content"

            email_item = {
                "id": msg['id'],
                "subject": subject, 
                "from": sender, 
                "body": body_data
            }
            
            email_list.append(email_item)

        except Exception as e:
            print(f"Email processing error: {e}")
            continue  

    # Update cache for full fetches
    if not check_new_only:
        email_cache = email_list
    
    return email_list

# Load pre-trained urgency classification model
ensemble_model = joblib.load("ensemble_model.pkl")

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()
keywords = ['urgent', 'critical', 'asap', 'immediate', 'important', 'immediately', 'as soon as possible', 
            'please reply', 'need response', 'emergency', 'high priority', 'time-sensitive', 'priority', 
            'top priority', 'urgent matter', 'respond quickly', 'time-critical', 'pressing', 'crucial', 
            'respond promptly', 'without delay']

def create_inference_df(subject, body):
    # Feature extraction for urgency prediction
    text = subject + " " + body
    processed_text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])
    keyword_feature = sum(1 for word in processed_text.split() if word in keywords)
    sentiment_score = (analyzer.polarity_scores(processed_text)['compound'] + 1) / 2

    print("\n--- Feature Extraction Debug ---")
    print(f"Subject: {subject}")
    print(f"Body: {body[:100]}...")
    print(f"Processed Text: {processed_text[:100]}...")
    print(f"Keyword Feature: {keyword_feature}")
    print(f"Sentiment Score: {sentiment_score}")
    
    df_infer = pd.DataFrame([[processed_text, keyword_feature, sentiment_score]],
                            columns=['processed_text', 'keyword_feature', 'sentiment'])
    return df_infer


def classify_urgency(email_data):
    # Predict and group urgency levels
    df_infer = create_inference_df(email_data["subject"], email_data["body"])
    prediction = ensemble_model.predict(df_infer)[0]
    
    if prediction in [3, 2]:
        return "Non-Urgent"
    elif prediction in [1, 0]:
        return "Urgent"
    else:
        return "Unknown"


def urgency_color(urgency_label):
    # Map urgency labels to UI colors
    color_map = {
        "Urgent": "red",
        "Non-Urgent": "blue",
        "Unknown": "gray"
    }
    return color_map.get(urgency_label, "black")


def process_emails(emails):
    # Add urgency classification to emails
    for email_item in emails:
        label = classify_urgency(email_item)
        email_item["urgency_label"] = label
        email_item["urgency_color"] = urgency_color(label)
    return emails


def email_checker_thread():
    # Background thread for monitoring new emails
    while True:
        try:
            new_emails = getEmails(check_new_only=True)
            if new_emails:
                processed_emails = process_emails(new_emails)
                email_cache.extend(processed_emails)
                
                for email in processed_emails:
                    new_email_queue.put(email)
                
                global waiting_clients
                for client in waiting_clients[:]:
                    try:
                        client.put(processed_emails)
                    except:
                        pass
                
                waiting_clients = []
            
            time.sleep(2)
        except Exception as e:
            print(f"Email checker thread error: {e}")
            time.sleep(5)


@app.route('/stream-emails')
def stream_emails():
    # Server-sent events endpoint for real-time updates
    def generate():
        client_queue = queue.Queue()
        waiting_clients.append(client_queue)
        
        yield "data: {}\n\n"
        
        while True:
            try:
                emails = client_queue.get(timeout=30)
                yield f"data: {json.dumps({'emails': emails})}\n\n"
                if client_queue in waiting_clients:
                    waiting_clients.remove(client_queue)
                break
            except queue.Empty:
                yield "data: {}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")


@app.route('/')
def index():
    # Main application route
    emails = getEmails()
    processed_emails = process_emails(emails)

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Urgency Classifier</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            // Email UI management variables
            let emailContainer;
            const displayedEmailIds = new Set();
            let eventSource;
            
            // Initialize event streaming when page loads
            document.addEventListener('DOMContentLoaded', function() {
                emailContainer = document.getElementById('email-container');
                
                document.querySelectorAll('.email-card').forEach(card => {
                    if (card.dataset.emailId) {
                        displayedEmailIds.add(card.dataset.emailId);
                    }
                });
                
                connectEventSource();
            });
            
            // Connect to server events stream
            function connectEventSource() {
                if (eventSource) {
                    eventSource.close();
                }
                
                eventSource = new EventSource('/stream-emails');
                
                eventSource.onmessage = function(event) {
                    if (event.data && event.data.trim() !== '{}') {
                        try {
                            const data = JSON.parse(event.data);
                            if (data.emails && data.emails.length > 0) {
                                addNewEmailsToUI(data.emails);
                                showNotification(data.emails.length);
                                connectEventSource();
                            }
                        } catch (e) {
                            console.error('Error parsing event data:', e);
                        }
                    }
                };
                
                eventSource.onerror = function() {
                    console.log('EventSource connection error, reconnecting...');
                    eventSource.close();
                    setTimeout(connectEventSource, 3000);
                };
            }
            
            // Add new emails to the interface
            function addNewEmailsToUI(emails) {
                emails.forEach(email => {
                    if (displayedEmailIds.has(email.id)) {
                        return;
                    }
                    
                    displayedEmailIds.add(email.id);
                    
                    const emailCard = document.createElement('div');
                    emailCard.className = 'email-card border border-gray-200 p-4 mb-2 rounded-xl shadow-md bg-white cursor-pointer hover:shadow-lg hover:bg-gray-100 transition duration-300';
                    emailCard.dataset.urgency = email.urgency_label;
                    emailCard.dataset.emailId = email.id;
                    emailCard.style.setProperty('--urgency-border', email.urgency_color);
                    emailCard.onclick = function() {
                        openModal(email.subject, email.from, email.body);
                    };
                    
                    emailCard.innerHTML = `
                        <div class="flex flex-col space-y-1 w-full">
                            <div class="font-semibold text-lg text-gray-900 truncate">${email.subject}</div>
                            <div class="text-gray-500 text-sm">From: ${email.from}</div>
                            <div class="text-gray-600 text-sm mt-1 line-clamp-2">
                                ${email.body.split(' ').slice(0, 80).join(' ')}${email.body.split(' ').length > 80 ? '...' : ''}
                            </div>
                        </div>
                        <span class="px-3 py-1 text-xs font-semibold rounded-full shadow-sm"
                            style="background-color: ${email.urgency_color}; color: white;">
                            ${email.urgency_label}
                        </span>
                    `;
                    
                    emailCard.style.opacity = '0';
                    emailCard.style.transform = 'translateY(-20px)';
                    emailContainer.insertBefore(emailCard, emailContainer.firstChild);
                    
                    setTimeout(() => {
                        emailCard.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                        emailCard.style.opacity = '1';
                        emailCard.style.transform = 'translateY(0)';
                    }, 10);
                });
            }
            
            // Display notification for new emails
            function showNotification(count) {
                const notification = document.createElement('div');
                notification.className = 'fixed top-4 right-4 bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg';
                notification.textContent = `${count} new email${count === 1 ? '' : 's'} received`;
                
                document.body.appendChild(notification);
                
                setTimeout(() => {
                    notification.style.opacity = '0';
                    notification.style.transition = 'opacity 0.5s ease';
                    setTimeout(() => notification.remove(), 500);
                }, 3000);
            }

            // Filter emails by urgency
            function filterEmails(urgency) {
                document.querySelectorAll('.email-card').forEach(card => {
                    if (urgency === 'All' || card.dataset.urgency === urgency) {
                        card.style.display = 'block';
                    } else {
                        card.style.display = 'none';
                    }
                });
            }

            // Open email detail modal
            function openModal(subject, sender, body) {
                document.getElementById("modal-subject").textContent = subject;
                document.getElementById("modal-sender").textContent = sender;
                document.getElementById("modal-body").textContent = body;
                document.getElementById("email-modal").classList.remove("hidden");
            }

            // Close email detail modal
            function closeModal() {
                document.getElementById("email-modal").classList.add("hidden");
            }

            // Send reply to email
            function sendReply() {
                const replyText = document.getElementById("reply-text").value;
                if (replyText.trim() === "") {
                    alert("Reply cannot be empty.");
                    return;
                }
                alert("Reply Sent: " + replyText);
                document.getElementById("reply-text").value = "";
                closeModal();
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
            <div class="mb-4 flex justify-between items-center">
                <div>
                    <label class="font-medium">Filter by Urgency:</label>
                    <select class="border p-2 rounded" onchange="filterEmails(this.value)">
                        <option value="All">All</option>
                        <option value="Urgent">Urgent</option>
                        <option value="Non-Urgent">Non-Urgent</option>
                    </select>
                </div>
                <div class="text-sm text-gray-500">
                    <span>Real-time email updates</span>
                </div>
            </div>        
            <div id="email-container">
                {% for email in emails %}
                    <div class="email-card border border-gray-200 p-4 mb-2 rounded-xl shadow-md bg-white cursor-pointer 
                        hover:shadow-lg hover:bg-gray-100 transition duration-300"
                        data-urgency="{{ email.urgency_label }}"
                        data-email-id="{{ email.id }}"
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
        </div>

        <!-- Email detail modal -->
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

                <!-- Reply composition area -->
                <div class="border rounded-lg mt-4 shadow-sm">
                    <textarea id="reply-text" class="w-full p-3 border-none focus:outline-none resize-none" placeholder="Type your reply..." rows="4"></textarea>

                    <!-- Formatting toolbar -->
                    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" />

                    <div class="flex items-center justify-between p-2 border-t bg-gray-100">
                        <div class="flex space-x-3 text-gray-600">
                            <!-- Formatting options -->
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Formatting Options">
                                <span class="material-symbols-outlined">format_color_text</span>
                            </button>
                            
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Attach Files">
                                <span class="material-symbols-outlined">attach_file</span>
                            </button>
                            
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Insert Link">
                                <span class="material-symbols-outlined">link</span>
                            </button>
                            
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Insert Emoji">
                                <span class="material-symbols-outlined">mood</span>
                            </button>
                            
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Insert Files Using Drive">
                                <span class="material-symbols-outlined">drive_export</span>
                            </button>
                            
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Insert Photo">
                                <span class="material-symbols-outlined">image</span>
                            </button>
                            
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Toggle Confidential Mode">
                                <span class="material-symbols-outlined">lock_clock</span>
                            </button>
                            
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="Insert Signature">
                                <span class="material-symbols-outlined">draw</span>
                            </button>
                            
                            <button class="hover:text-blue-500 focus:outline-none" aria-label="More Options">
                                <span class="material-symbols-outlined">more_vert</span>
                            </button>
                        </div>

                        <!-- Send button -->
                        <button onclick="sendReply()" class="bg-blue-500 text-white px-4 py-2 rounded-lg shadow-md hover:bg-blue-600 focus:outline-none">
                            Send
                        </button>
                    </div>

                </div>
            </div>
        </div>
    </body>
    <style>
    /* Custom styling for urgency indicators */
        .email-card {
            border-color: var(--urgency-border, #e5e7eb);
            transition: border-color 0.3s ease-in-out;
        }

        .email-card:hover {
            border-color: var(--urgency-border);
        }
        
        /* Animation for newly arrived emails */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .new-email {
            animation: fadeInDown 0.5s ease forwards;
        }
    </style>

    </html>
    """
    return render_template_string(html_template, emails=processed_emails)

if __name__ == '__main__':
    # Start background email monitoring
    email_thread = threading.Thread(target=email_checker_thread, daemon=True)
    email_thread.start()
    
    app.run(debug=True)