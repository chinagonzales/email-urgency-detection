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

# Ensure necessary nltk resources are available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df[['title', 'body', 'urgency']]
    df.dropna(inplace=True) 
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
    class_counts = Counter(y_train)
    vocab = set()
    word_counts = defaultdict(lambda: defaultdict(int))

    # Count words per class
    for text, label in zip(X_train, y_train):
        tokens = preprocess_text(text)
        for token in tokens:
            vocab.add(token)
            word_counts[label][token] += 1

    # Compute probabilities
    class_priors = {cls: np.log(count / len(y_train)) for cls, count in class_counts.items()}
    word_probs = {}

    for cls in class_counts:
        total_words = sum(word_counts[cls].values())
        word_probs[cls] = {word: np.log((word_counts[cls][word] + 1) / (total_words + len(vocab))) for word in vocab}

    return class_priors, word_probs, vocab

# Predict function
def predict(text, class_priors, word_probs, vocab):
    tokens = preprocess_text(text)
    scores = {}

    for cls, prior in class_priors.items():
        scores[cls] = prior
        total_words = sum(np.exp(list(word_probs[cls].values())))
        unseen_prob = np.log(1 / (total_words + len(vocab)))

        for token in tokens:
            if token in vocab:
                scores[cls] += word_probs[cls].get(token, unseen_prob)
            else:
                scores[cls] += unseen_prob

    return max(scores, key=scores.get)

# Evaluate model
def evaluate_mnb(X_test, y_test, class_priors, word_probs, vocab, model_name="Multinomial Naive Bayes"):
    start_time = time.time()
    y_pred = [predict(text, class_priors, word_probs, vocab) for text in X_test]
    end_time = time.time()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    time_taken = end_time - start_time
    
    print(f"Evaluation for {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Time Taken: {time_taken:.4f} seconds\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    return {"accuracy": accuracy, "precision": precision, "time_taken": time_taken}

# Main execution
df = load_dataset("all_tickets.csv")

# Combine title and body into a single text feature
df['text'] = df['title'] + " " + df['body']
X = df['text'].tolist()
y = df['urgency'].tolist()

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training size after split: {len(X_train)} samples")
print(f"Testing size after split: {len(X_test)} samples")

# Apply Random Over Sampling to balance the dataset
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(pd.DataFrame(X_train), y_train)
X_train_resampled = X_train_resampled[0].tolist() 

# Train and evaluate
train_start_time = time.time()
class_priors, word_probs, vocab = train_mnb(X_train_resampled, y_train_resampled)
train_end_time = time.time()

evaluation_results = evaluate_mnb(X_test, y_test, class_priors, word_probs, vocab)
