import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')


# Enhanced Multinomial Naive Bayes built from scratch with unified TF-IDF
class EnhancedMNB(BaseEstimator, ClassifierMixin):
    def __init__(self, ngram_range=(1, 2), alpha=1.0):
        self.ngram_range = ngram_range
        self.alpha = alpha  # Laplace smoothing parameter
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        self.class_priors = None
        self.feature_probs = None
        self.classes_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            text_data = X['processed_text'].values.tolist()
            tfidf_matrix = self.vectorizer.fit_transform(text_data)
            additional_features = csr_matrix(X[['keyword_feature', 'sentiment']].values)
            X = hstack([tfidf_matrix, additional_features]).tocsr()
        elif not isinstance(X, csr_matrix):
            raise ValueError("X must be a pandas DataFrame or a sparse CSR matrix.")

        # Initialize class-related parameters
        n_classes = len(np.unique(y))
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.class_priors = np.zeros(n_classes)
        self.feature_probs = np.zeros((n_classes, n_features))

        for c_idx, c_label in enumerate(self.classes_):
            indices = np.where(y == c_label)[0]
            X_c = X[indices]

            # Class prior and feature probabilities with Laplace smoothing
            self.class_priors[c_idx] = X_c.shape[0] / X.shape[0]
            self.feature_probs[c_idx, :] = (X_c.sum(axis=0) + self.alpha) / (
                X_c.sum() + self.alpha * n_features
            )

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            text_data = X['processed_text'].values.tolist()
            tfidf_matrix = self.vectorizer.transform(text_data)
            additional_features = csr_matrix(X[['keyword_feature', 'sentiment']].values)
            X = hstack([tfidf_matrix, additional_features]).tocsr()
        elif not isinstance(X, csr_matrix):
            raise ValueError("X must be a pandas DataFrame or a sparse CSR matrix.")

        log_probs = np.log(self.class_priors) + X @ np.log(self.feature_probs.T)
        return self.classes_[np.argmax(log_probs, axis=1).ravel()]

    def predict_proba(self, X):
        X_transformed = self._prepare_features(X)
        log_probs = np.log(self.class_priors) + X_transformed @ np.log(self.feature_probs.T)
        exp_probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        return exp_probs / np.sum(exp_probs, axis=1, keepdims=True)

    def _prepare_features(self, X):
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            if 'processed_text' not in X.columns:
                raise ValueError("DataFrame must contain a 'processed_text' column.")
            return self.vectorizer.transform(X['processed_text'].values)

        # Handle list or 1D array
        elif isinstance(X, (list, np.ndarray)):
            return self.vectorizer.transform(X)

        # Handle sparse matrices directly
        elif isinstance(X, scipy.sparse.spmatrix):
            return X

        else:
            raise ValueError(f"Unexpected data format for features: {type(X)}")

# Load Dataset
df = pd.read_csv('all_tickets.csv')

# Handle missing values
df.dropna(subset=['body', 'title', 'urgency'], inplace=True)

# Ensure urgency column is integer
df['urgency'] = df['urgency'].astype(int)

# Combine 'body' and 'title' for text processing
df['text'] = df['title'] + ' ' + df['body']

# Tokenization and Lemmatization
lemmatizer = WordNetLemmatizer()
df['processed_text'] = df['text'].apply(
    lambda text: ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)])
)

# Keyword-based feature extraction
keywords = ['urgent', 'critical', 'asap', 'important', 'immediate']
def keyword_feature(text):
    return sum(1 for word in text.split() if word in keywords)
df['keyword_feature'] = df['processed_text'].apply(keyword_feature)

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['processed_text'].apply(lambda x: (analyzer.polarity_scores(x)['compound'] + 1) / 2)

# Feature and Label Split
X = df[['processed_text', 'keyword_feature', 'sentiment']]
y = df['urgency']

# Print class distribution BEFORE resampling
print("Class Distribution Before Resampling:")
print(y.value_counts())

# Apply oversampling to balance class distribution
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Print class distribution AFTER resampling
print("Class Distribution After Resampling:")
print(pd.Series(y_resampled).value_counts())  # Check if classes are now balanced

# Convert back to DataFrame
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Vectorize and combine features for train and test sets
def prepare_combined_features(X_train, X_test, vectorizer):
    tfidf_matrix_train = vectorizer.fit_transform(X_train['processed_text'])
    tfidf_matrix_test = vectorizer.transform(X_test['processed_text'])

    additional_features_train = csr_matrix(X_train[['keyword_feature', 'sentiment']].values)
    additional_features_test = csr_matrix(X_test[['keyword_feature', 'sentiment']].values)

    X_train_combined = hstack([tfidf_matrix_train, additional_features_train]).tocsr()
    X_test_combined = hstack([tfidf_matrix_test, additional_features_test]).tocsr()

    return X_train_combined, X_test_combined

X_train_combined, X_test_combined = prepare_combined_features(X_train, X_test, vectorizer)

# Feature Selection
k_best = SelectKBest(chi2, k=500)
X_train_selected = k_best.fit_transform(X_train_combined, y_train)
X_test_selected = k_best.transform(X_test_combined)

# Grid Search for Enhanced MNB
param_grid = {'alpha': [0.1, 0.5, 1.0]}
grid_search = GridSearchCV(EnhancedMNB(), param_grid, scoring='accuracy', cv=5, n_jobs=1)

# Train Enhanced MNB
start_time = time.time()
grid_search.fit(X_train_selected, y_train)
mnb_time = time.time() - start_time

# Best Model
mnb_model = grid_search.best_estimator_

# Multinomial Logistic Regression
log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
start_time = time.time()
log_reg.fit(X_train_combined, y_train)
log_reg_time = time.time() - start_time

# Ensemble Model
ensemble = VotingClassifier(estimators=[('enhanced_mnb', mnb_model), ('log_reg', log_reg)], voting='soft')
ensemble.fit(X_train_selected, y_train)
ensemble_time = time.time() - start_time

# Evaluation Functions
def evaluate_model(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print(f"{title} Results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{title} - Confusion Matrix')
    plt.show()

# Evaluate Models
evaluate_model(mnb_model, X_test_selected, y_test, "Enhanced MNB")
evaluate_model(log_reg, X_test_combined, y_test, "Multinomial Logistic Regression")
evaluate_model(ensemble, X_test_selected, y_test, "Ensemble Model")

# Save trained models
joblib.dump(mnb_model, 'enhanced_mnb.pkl')
joblib.dump(log_reg, 'logistic_reg.pkl')
joblib.dump(ensemble, 'ensemble_model.pkl')

# Save preprocessing objects
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(k_best, 'feature_selector.pkl')

print("Models and preprocessing objects saved successfully!")