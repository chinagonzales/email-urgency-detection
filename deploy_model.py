import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from enhanced_mnb import EnhancedMNB

# Load trained models
mnb_model = joblib.load('enhanced_mnb.pkl')
log_reg = joblib.load('logistic_reg.pkl')
ensemble = joblib.load('ensemble_model.pkl')

# Load vectorizer and feature selector
vectorizer = joblib.load('tfidf_vectorizer.pkl')
k_best = joblib.load('feature_selector.pkl')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Keyword list for urgency detection
keywords = ['urgent', 'critical', 'asap', 'important', 'immediate']

def keyword_feature(text):
    """Extract keyword-based urgency feature."""
    return sum(1 for word in text.split() if word in keywords)

def preprocess_text(text):
    """Tokenize, lemmatize, and extract sentiment and keyword features."""
    processed_text = ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)])
    keyword_score = keyword_feature(processed_text)
    sentiment_score = (analyzer.polarity_scores(processed_text)['compound'] + 1) / 2
    return processed_text, keyword_score, sentiment_score

def predict_urgency(text):
    """Make a prediction for a new text sample."""
    processed_text, keyword_score, sentiment_score = preprocess_text(text)

    # Debugging prints
    print(f"\n🔍 Debugging for input: {text}")
    print(f"Processed Text: {processed_text}")
    print(f"Keyword Score: {keyword_score}")
    print(f"Sentiment Score: {sentiment_score}")

    # Convert text to TF-IDF
    tfidf_vector = vectorizer.transform([processed_text])
    additional_features = csr_matrix([[keyword_score, sentiment_score]])

    # Combine TF-IDF with additional features
    final_features = hstack([tfidf_vector, additional_features])
    final_features = k_best.transform(final_features)

    # Print extracted TF-IDF values (only if nonzero)
    print(f"TF-IDF Feature Sum: {final_features.sum()}") 

    # Predict using probabilities
    probabilities = ensemble.predict_proba(final_features)[0]

    # Debugging probabilities
    print(f"Probabilities: {probabilities}")  

    # Apply custom threshold: default to class 2 if uncertain
    prediction = np.argmax(probabilities) if max(probabilities) > 0.5 else 0

    print(f"Final Prediction: {prediction}\n")

    return prediction

# Example usage
if __name__ == "__main__":
    sample_text = "Urgent! Fix it ASAP."
    urgency_level = predict_urgency(sample_text)
    print(f"Predicted Urgency Level: {urgency_level}")
