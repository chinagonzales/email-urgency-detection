# ensemble.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import scipy.sparse
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import time
import joblib
import nltk

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# --- Enhanced Multinomial Naive Bayes built from scratch --- #
class EnhancedMNB(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, k_best=None, ngram_range=(1, 1)):
        self.alpha = alpha             
        self.k_best = k_best           
        self.ngram_range = ngram_range 
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        self.selector = None           
        self.classes_ = None           
        self.class_log_prior = None
        self.feature_log_prob = None
        self._estimator_type = "classifier"

    def _combine_features(self, X):
        """Combine text TF-IDF with numeric features."""
        X_text = X["processed_text"].values
        # Fit vectorizer if not fitted
        if not hasattr(self.vectorizer, "vocabulary_"):
            self.vectorizer.fit(X_text)  
        X_tfidf = self.vectorizer.transform(X_text)
        keyword_feat = csr_matrix(X[['keyword_feature']].values.astype(float))
        sentiment_feat = csr_matrix(X[['sentiment']].values.astype(float))
        return hstack([X_tfidf, keyword_feat, sentiment_feat])

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_combined = self._combine_features(X)
        else:
            raise ValueError("X must be a DataFrame for training.")

        # Apply feature selection if requested
        if self.k_best and self.k_best > 0:
            self.selector = SelectKBest(chi2, k=min(self.k_best, X_combined.shape[1]))
            X_selected = self.selector.fit_transform(X_combined, y)
        else:
            self.selector = None
            X_selected = X_combined

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        num_features = X_selected.shape[1]
        self.feature_log_prob = np.zeros((num_classes, num_features))
        self.class_log_prior = np.zeros(num_classes)

        for i, c in enumerate(self.classes_):
            mask = (y == c)
            X_c = X_selected[mask]
            self.class_log_prior[i] = np.log(X_c.shape[0] / X_selected.shape[0])
            feature_counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob[i, :] = np.log(feature_counts / feature_counts.sum())

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_combined = self._combine_features(X)
        elif isinstance(X, scipy.sparse.spmatrix):
            X_combined = X
        else:
            raise ValueError("Unexpected input type for transform()")

        if self.selector is not None:
            X_selected = self.selector.transform(X_combined)
        else:
            X_selected = X_combined

        return X_selected

    def predict(self, X):
        X_selected = self.transform(X)
        if X_selected.shape[1] != self.feature_log_prob.shape[1]:
            raise ValueError(
                f"Mismatch: Model expects {self.feature_log_prob.shape[1]} features, "
                f"but got {X_selected.shape[1]}"
            )
        log_probs = X_selected @ self.feature_log_prob.T + self.class_log_prior
        return self.classes_[np.argmax(log_probs, axis=1)]

    def predict_proba(self, X):
        X_selected = self.transform(X)
        if X_selected.shape[1] != self.feature_log_prob.shape[1]:
            raise ValueError(
                f"Feature mismatch: Expected {self.feature_log_prob.shape[1]} features, "
                f"but got {X_selected.shape[1]}"
            )
        log_probs = X_selected @ self.feature_log_prob.T + self.class_log_prior
        return np.exp(log_probs) / np.exp(log_probs).sum(axis=1, keepdims=True)
        
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# --- Data Preparation ---
df = pd.read_csv('all_tickets.csv')
df.dropna(subset=['body', 'title', 'urgency'], inplace=True)
df['urgency'] = df['urgency'].astype(int)
df['text'] = df['title'] + ' ' + df['body']

lemmatizer = WordNetLemmatizer()
df['processed_text'] = df['text'].apply(
    lambda text: ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)])
)

# Keyword-based feature extraction
keywords = ['urgent', 'critical', 'asap', 'immediate', 'important', 'immediately', 
            'as soon as possible','please reply', 'need response', 'emergency', 'high priority']
def keyword_feature(text):
    return sum(1 for word in text.split() if word in keywords)
df['keyword_feature'] = df['processed_text'].apply(keyword_feature)

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
def compute_sentiment(text):
    vs = analyzer.polarity_scores(text)
    return (vs['compound'] + 1) / 2
df['sentiment'] = df['processed_text'].apply(compute_sentiment)

X = df[['processed_text', 'keyword_feature', 'sentiment']]
y = df['urgency']

print("Class Distribution Before Resampling:")
print(y.value_counts())

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
print("Class Distribution After Resampling:")
print(pd.Series(y_resampled).value_counts())
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

def train_and_save_models():
    # --- EnhancedMNB with Grid Search ---
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0],
        'ngram_range': [(1, 1), (1, 2)],
        'k_best': [100, 500, 1000]
    }
    mnb_model = EnhancedMNB()
    grid_search = GridSearchCV(mnb_model, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)
    grid_search.fit(X_train, np.array(y_train))
    print("Best parameters found for EnhancedMNB:", grid_search.best_params_)
    best_mnb_model = grid_search.best_estimator_

    # --- Logistic Regression ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(ngram_range=(1, 2)), 'processed_text'),
        ]
    )
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('log_reg', LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)

    # --- Ensemble ---
    ensemble = VotingClassifier(estimators=[
        ('enhanced_mnb', best_mnb_model),
        ('lr_pipeline', lr_pipeline)
    ], voting='soft')

    print("Performing Cross-Validation on the Ensemble Model...")
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
    print("CV Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", np.mean(cv_scores))

    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, 'ensemble_model.pkl')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_models()
