import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure necessary nltk resources are available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Keyword-based feature extraction
keywords = ['urgent', 'critical', 'asap', 'immediate', 'important', 'immediately', 'as soon as possible', 
            'please reply', 'need response', 'emergency', 'high priority', 'time-sensitive', 'priority', 
            'top priority', 'urgent matter', 'respond quickly', 'time-critical', 'pressing', 'crucial', 
            'respond promptly', 'without delay']

def keyword_feature(text):
    return sum(1 for word in text.split() if word in keywords) + text.count('!')

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
def compute_sentiment(text):
    vs = analyzer.polarity_scores(text)
    return (vs['compound'] + 1) / 2

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df[['title', 'body', 'urgency']]
    df.dropna(inplace=True)
    df['text'] = df['title'] + " " + df['body']
    df['keyword_feature'] = df['text'].apply(keyword_feature)
    df['sentiment'] = df['text'].apply(compute_sentiment)
    return df

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]

# Train Multinomial Naive Bayes
def train_mnb(X_train, y_train):
    start_time = time.time()
    class_counts = Counter(y_train)
    vocab = set()
    word_counts = defaultdict(lambda: defaultdict(int))

    for text, label in zip(X_train['text'], y_train):
        tokens = preprocess_text(text)
        for token in tokens:
            vocab.add(token)
            word_counts[label][token] += 1

    class_priors = {cls: np.log(count / len(y_train)) for cls, count in class_counts.items()}
    word_probs = {}

    for cls in class_counts:
        total_words = sum(word_counts[cls].values())
        word_probs[cls] = {word: np.log((word_counts[cls][word] + 1) / (total_words + len(vocab))) for word in vocab}

    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training Time: {training_time:.4f} seconds\n")
    return class_priors, word_probs, vocab, training_time

# Predict function
def predict(row, class_priors, word_probs, vocab):
    tokens = preprocess_text(row['text'])
    scores = {}
    for cls, prior in class_priors.items():
        scores[cls] = prior
        total_words = sum(np.exp(list(word_probs[cls].values())))
        unseen_prob = np.log(1 / (total_words + len(vocab)))
        for token in tokens:
            scores[cls] += word_probs[cls].get(token, unseen_prob)
        scores[cls] += np.log(1 + row['keyword_feature'])
        scores[cls] += np.log(1 + row['sentiment'])
    return max(scores, key=scores.get)

# Evaluate model
def evaluate_mnb(X_test, y_test, class_priors, word_probs, vocab, training_time, model_name="Multinomial Naive Bayes"):
    start_time = time.time()
    y_pred = [predict(row, class_priors, word_probs, vocab) for _, row in X_test.iterrows()]
    end_time = time.time()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    prediction_time = end_time - start_time
    
    print(f"Evaluation for {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Prediction Time: {prediction_time:.4f} seconds\n")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "training_time": training_time,
        "prediction_time": prediction_time
    }

# Main execution
df = load_dataset("all_tickets.csv")
X = df[['text', 'keyword_feature', 'sentiment']]
y = df['urgency']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

class_priors, word_probs, vocab, training_time = train_mnb(X_train_resampled, y_train_resampled)
evaluation_results = evaluate_mnb(X_test, y_test, class_priors, word_probs, vocab, training_time)
