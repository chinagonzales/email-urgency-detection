from flask import Flask, request, jsonify
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from enhanced_mnb import EnhancedMNB

app = Flask(__name__)

# Load trained models
ensemble = joblib.load('ensemble_model.pkl', mmap_mode=None)
vectorizer = joblib.load('tfidf_vectorizer.pkl')
k_best = joblib.load('feature_selector.pkl')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Keywords for urgency detection
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

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for real-time urgency prediction."""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    processed_text, keyword_score, sentiment_score = preprocess_text(text)

    # Debugging print statements
    print(f"Received Text: {text}")
    print(f"Processed Text: {processed_text}")
    print(f"Keyword Score: {keyword_score}")
    print(f"Sentiment Score: {sentiment_score}")

    # Convert text to TF-IDF
    tfidf_vector = vectorizer.transform([processed_text])
    additional_features = csr_matrix([[keyword_score, sentiment_score]])

    # Combine TF-IDF with additional features
    final_features = hstack([tfidf_vector, additional_features])
    final_features = k_best.transform(final_features)

    # Predict using the ensemble model
    prediction = ensemble.predict(final_features)[0]

    print(f"Final Prediction: {prediction}")

    return jsonify({'urgency_level': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import psutil
print(f"Available Memory: {psutil.virtual_memory().available / (1024 * 1024)} MB")