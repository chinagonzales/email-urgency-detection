import time
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_distances
import gc

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Enhanced kNN Classifier (with TF-IDF)
class EnhancedKNN(BaseEstimator, ClassifierMixin):
    
    def __init__(self, k=5, ngram_range=(1, 1), max_features=5000):
        """ Initialize the EnhancedKNN class with parameters for k-nearest neighbors, 
            n-gram range for TfidfVectorizer, and maximum number of features for TfidfVectorizer."""
        self.k = k
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        self.X_train_combined = None
        self.y_train = None
        self.classes_ = None

    def get_params(self, deep=True):
        # Get parameters for GridSearchCV.
        return {"k": self.k, "ngram_range": self.ngram_range, "max_features": self.max_features}

    def set_params(self, **params):
        # Set parameters for GridSearchCV.
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _combine_features(self, X, fit_vectorizer=False):
        # Convert text and numerical features into a sparse matrix.
        """ This method transforms the input text data into TF-IDF features, and combines it 
            with keyword and sentiment features into a single sparse matrix for efficient processing."""
        
        X_text = X["processed_text"].values

        if fit_vectorizer:
            # Fit and transform the vectorizer on training data.
            X_tfidf = self.vectorizer.fit_transform(X_text)
        else:
            # Transform the text data using the fitted vectorizer.
            X_tfidf = self.vectorizer.transform(X_text)

        # Convert keyword and sentiment features to sparse matrices.
        keyword_feat = csr_matrix(X[['keyword_feature']].values.astype(float))
        sentiment_feat = csr_matrix(X[['sentiment']].values.astype(float))

        # Weight the keyword feature by multiplying by 2.
        keyword_feat *= 2

        # Ensure sentiment feature values are at least 0.5.
        sentiment_feat.data = np.where(sentiment_feat.data < 0.5, 0.5, sentiment_feat.data)

        # Combine TF-IDF, keyword, and sentiment features into one sparse matrix.
        return hstack([X_tfidf, keyword_feat, sentiment_feat])

    def fit(self, X, y):
        # Fit the kNN model with transformed data.
        """This method processes the training data by combining its features and fitting the model."""
        X_combined = self._combine_features(X, fit_vectorizer=True)
        self.X_train_combined = X_combined
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)  # Set the classes_ attribute for use in evaluation.

    def predict(self, X):
        # Predict labels for the test set using the kNN model.
        """This method processes the test data in batches to manage memory usage, calculates 
            distances between test instances and training instances, and predicts labels based 
            on the k-nearest neighbors. """
        
        batch_size = 100  # Process data in smaller batches to manage memory usage.
        y_pred = []

        for i in range(0, X.shape[0], batch_size):
            # Process each batch of data.
            X_batch = X.iloc[i:i+batch_size]
            X_test_combined = self._combine_features(X_batch, fit_vectorizer=False)
            
            # Calculate distances between test instances and training instances.
            distances = pairwise_distances(X_test_combined, self.X_train_combined, metric='euclidean')
            
            # Identify the indices of the k-nearest neighbors.
            nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
            
            # Retrieve the labels of the k-nearest neighbors.
            nearest_labels = self.y_train[nearest_indices]

            # Use a simple majority vote to predict the label for each instance.
            y_batch_pred = [Counter(nearest_labels[j]).most_common(1)[0][0] for j in range(nearest_labels.shape[0])]
            y_pred.extend(y_batch_pred)
            gc.collect()  # Run garbage collection to free up memory.

        return np.array(y_pred)

# Evaluation function to assess the model's performance.
def evaluate_model(model, X_test, y_test, model_name):
    # Start the timer to measure prediction time.
    start_time = time.time()
    y_pred = model.predict(X_test)  # Predict the labels for the test set.
    end_time = time.time()  # End the timer.
    
    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy.
    precision = precision_score(y_test, y_pred, average='weighted')  # Calculate the precision.
    cm = confusion_matrix(y_test, y_pred)  # Compute the confusion matrix.
    time_taken = end_time - start_time  # Calculate the time taken for prediction.
    
    # Display the evaluation metrics.
    print(f"{model_name} - Accuracy: {accuracy:.4f}")
    print(f"{model_name} - Precision: {precision:.4f}")
    print(f"{model_name} - Time Taken: {time_taken:.4f} seconds\n")
    print("Confusion Matrix:")
    print(cm)
    
    # Plot the confusion matrix.
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

# Load data from CSV file.
df = pd.read_csv('all_tickets.csv')
df.dropna(subset=['body', 'title', 'urgency'], inplace=True)  # Drop rows with missing values in specified columns.
df['urgency'] = df['urgency'].astype(int)  # Convert 'urgency' column to integer type.
df['text'] = df['title'] + ' ' + df['body']  # Combine 'title' and 'body' columns into a single text column.

# Text processing with lemmatization.
lemmatizer = WordNetLemmatizer()
df['processed_text'] = df['text'].apply(
    lambda text: ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)])
)

# Keyword feature extraction.
""" Words or phrases that are likely to be important to determine the urgency of the ticket. 
These are manually chosen based on the context and possible urgency indicators. """
keywords = ['urgent', 'critical', 'asap', 'immediate', 'important', 'immediately',
            'as soon as possible', 'please reply', 'need response', 'emergency',
            'high priority', 'time-sensitive', 'priority', 'top priority', 'urgent matter',
            'respond quickly', 'time-critical', 'pressing', 'crucial', 'respond promptly', 'without delay']
# Create a new feature 'keyword_feature' that counts the occurrence of these keywords in the processed text.
df['keyword_feature'] = df['processed_text'].apply(lambda text: sum(1 for word in text.split() if word in keywords) + text.count('!'))

# Sentiment analysis using VADER.
analyzer = SentimentIntensityAnalyzer()
# Create a new feature 'sentiment' that scores the sentiment of the processed text, scaled to a range [0, 1].
df['sentiment'] = df['processed_text'].apply(lambda text: (analyzer.polarity_scores(text)['compound'] + 1) / 2)

# Define features (X) and target variable (y).
X = df[['processed_text', 'keyword_feature', 'sentiment']]
y = df['urgency']

# Split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Resample the training data to address class imbalance.
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Convert resampled data back to DataFrame to preserve column names.
X_train = pd.DataFrame(X_train_resampled, columns=X.columns)
y_train = y_train_resampled

# Perform hyperparameter tuning using GridSearchCV.
param_grid = {
    'k': [3, 5, 7],  # Different values of k to try.
    'ngram_range': [(1, 1), (1, 2)],  # Different n-gram ranges to try.
    'max_features': [1000, 5000, 10000]  # Different values for max_features to try.
}
knn_model = EnhancedKNN()
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)
grid_search.fit(X_train, np.array(y_train))
print("Best parameters found for EnhancedKNN:", grid_search.best_params_)
best_knn_model = grid_search.best_estimator_

# Evaluate the best model found by GridSearchCV.
evaluate_model(best_knn_model, X_test, y_test, "Enhanced kNN")

# Run garbage collection to free up memory.
gc.collect()
