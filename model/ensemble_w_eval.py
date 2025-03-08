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
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.special import softmax
from sklearn.pipeline import Pipeline
import time
import joblib
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Enhanced Multinomial Naive Bayes
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
        X_text = X["processed_text"].values
        
        if not hasattr(self.vectorizer, "vocabulary_"):
            self.vectorizer.fit(X_text)
        X_tfidf = self.vectorizer.transform(X_text)
        
        keyword_feat = csr_matrix(X[['keyword_feature']].values.astype(float))
        sentiment_feat = csr_matrix(X[['sentiment']].values.astype(float))

        combined_features = hstack([X_tfidf, keyword_feat, sentiment_feat])

        return combined_features

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Expected DataFrame but got array. Ensure input remains a DataFrame.")
        X_combined = self._combine_features(X)

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
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Expected DataFrame but got array. Ensure input remains a DataFrame.")
        
        X_combined = self._combine_features(X)
        
        if self.selector:
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
    
class LogisticRegressionScratch(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, epochs=1000, tol=1e-4, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None

    def _initialize_parameters(self, n_features, n_classes):
        """Initialize weights and bias for multiclass classification."""
        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)

    def _compute_logits(self, X):
        """Compute linear combination (logits)."""
        if scipy.sparse.issparse(X):  
            return X @ self.weights.T + self.bias
        elif isinstance(X, pd.DataFrame):
            raise ValueError("LogisticRegressionScratch expects numerical features, not raw text DataFrame.")
        else:
            return np.dot(X, self.weights.T) + self.bias

    def _compute_loss(self, X, y, logits):
        """Compute categorical cross-entropy loss."""
        print(f"Debug: X shape in loss function: {X.shape}, y shape: {y.shape}")
        m = X.shape[0]
        print(f"logits shape: {logits.shape}")
        probs = softmax(logits, axis=1)
        print(f"probs shape: {probs.shape}")

        if y.ndim > 1:
            print(f"y shape: {y.shape}, unique values: {np.unique(y)}")
            y = np.argmax(y, axis=1)

        print(f"y shape: {y.shape}, y unique values: {np.unique(y)}")

        if probs.shape[0] != len(y):
            raise ValueError(f"Shape mismatch: probs shape {probs.shape}, y shape {y.shape}")

        correct_log_probs = -np.log(probs[np.arange(m), y])
        return np.mean(correct_log_probs)

    def _compute_gradients(self, X, y, probs):
        m = X.shape[0]
        y_one_hot = np.eye(self.weights.shape[0])[y] 
        grad_weights = (1/m) * ((probs - y_one_hot).T @ X.toarray()) + self.lambda_reg * self.weights
        grad_bias = (1/m) * np.sum(probs - y_one_hot, axis=0)
        return grad_weights, grad_bias

    def fit(self, X, y):
        """Train the model using gradient descent."""
        self.classes_ = np.unique(y) 
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        self._initialize_parameters(n_features, n_classes)

        prev_loss = float('inf')
        for epoch in range(self.epochs):
            logits = self._compute_logits(X)
            probs = softmax(logits, axis=1)
            loss = self._compute_loss(X, y, logits)
            grad_weights, grad_bias = self._compute_gradients(X, y, probs)

            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

            if abs(prev_loss - loss) < self.tol:
                break

            prev_loss = loss

        return self

    def predict_proba(self, X):
        """Predict probabilities for each class."""
        logits = self._compute_logits(X)
        return softmax(logits, axis=1)

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# Data Preparation
df = pd.read_csv('all_tickets.csv')
df.dropna(subset=['body', 'title', 'urgency'], inplace=True)
df['urgency'] = df['urgency'].astype(int)
df['text'] = df['title'] + ' ' + df['body']

lemmatizer = WordNetLemmatizer()
df['processed_text'] = df['text'].apply(
    lambda text: ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)])
)

# Keyword-based feature extraction
keywords = ['urgent', 'critical', 'asap', 'immediate', 'important', 'immediately', 'as soon as possible', 
            'please reply', 'need response', 'emergency', 'high priority', 'time-sensitive', 'priority', 
            'top priority', 'urgent matter', 'respond quickly', 'time-critical', 'pressing', 'crucial', 
            'respond promptly', 'without delay']
def keyword_feature(text):
    return sum(1 for word in text.split() if word in keywords) + text.count('!') 
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

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_test, y_test, model_name):
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    time_taken = end_time - start_time
    
    print(f"Evaluation for {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Time Taken: {time_taken:.4f} seconds\n")
    
    return {"accuracy": accuracy, "precision": precision, "time_taken": time_taken}

# Cross-validation 
def perform_cross_validation(model, X_train, y_train, model_name, cv_folds=5):
    print(f"Performing {cv_folds}-fold Cross-Validation for {model_name}...")
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores for {model_name}: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f} | Standard Deviation: {scores.std():.4f}\n")

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

def train_and_save_models(X_train, X_test, y_train, y_test):
    # Resampling
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    print("Class Distribution After Resampling:")
    print(pd.Series(y_train_resampled).value_counts())

    X_train = pd.DataFrame(X_train_resampled, columns=X.columns)
    y_train = y_train_resampled  

    for col in ['keyword_feature', 'sentiment']:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)

    # Train Enhanced MNB with full feature processing 
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

    # Cross-Validation for EnhancedMNB
    perform_cross_validation(best_mnb_model, X_train, y_train, "Enhanced MNB (with hyperparameter tuning)")

    # Evaluate MNB w/ hyperparameter tuning
    mnb_results = evaluate_model(best_mnb_model, X_test, y_test, "Enhanced MNB (with hyperparameter tuning)")

    # Plot Confusion Matrix for EnhancedMNB
    y_pred_mnb = best_mnb_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_mnb, "Enhanced MNB (with hyperparameter tuning)")

    # Logistic Regression
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train["processed_text"])
    X_test_tfidf = tfidf_vectorizer.transform(X_test["processed_text"])

    print("Training Logistic Regression (TF-IDF only)")
    log_reg_scratch = LogisticRegressionScratch(learning_rate=0.01, epochs=3000, lambda_reg=0.01)
    log_reg_scratch.fit(X_train_tfidf, y_train)  

    # Cross-Validation for Logistic Regression
    perform_cross_validation(log_reg_scratch, X_train_tfidf, y_train, "Logistic Regression")

    # Evaluate Logistic Regression
    lr_results = evaluate_model(log_reg_scratch, X_test_tfidf, y_test, "Logistic Regression")

    # Plot Confusion Matrix for Logistic Regression
    y_pred_log_reg = log_reg_scratch.predict(X_test_tfidf)
    plot_confusion_matrix(y_test, y_pred_log_reg, "Logistic Regression")

    # Pipelines for Ensemble Model 
    mnb_pipeline = Pipeline([
        ('mnb_feature_processing', best_mnb_model)
    ])

    logistic_pipeline = Pipeline([
        ('tfidf', ColumnTransformer(
            [('tfidf', TfidfVectorizer(), 'processed_text')],
            remainder='drop'  
        )),
        ('logreg', LogisticRegressionScratch(learning_rate=0.01, epochs=1000, lambda_reg=0.0001))
    ])

    # Ensemble
    ensemble = VotingClassifier(estimators=[
        ('enhanced_mnb', mnb_pipeline),      
        ('log_reg_scratch', logistic_pipeline)  
    ], voting='soft'
    )

    ensemble.fit(X_train, y_train)  

    # Cross-Validation for Ensemble
    perform_cross_validation(ensemble, X_train, y_train, "Ensemble Model")

    # Evaluate Ensemble 
    ensemble_results = evaluate_model(ensemble, X_test, y_test, "Ensemble Model")

    # Plot Confusion Matrix for Ensemble Model
    y_pred_ensemble = ensemble.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_ensemble, "Ensemble Model")

    # Save the model
    joblib.dump(ensemble, 'ensemble_model.pkl')
    print("Model saved successfully!")

    # Final Model Evaluations
    results_df = pd.DataFrame({
        'Model': ['Enhanced MNB', 'Logistic Regression Scratch', 'Ensemble'],
        'Accuracy': [mnb_results['accuracy'], lr_results['accuracy'], ensemble_results['accuracy']],
        'Precision': [mnb_results['precision'], lr_results['precision'], ensemble_results['precision']],
        'Time Taken': [mnb_results['time_taken'], lr_results['time_taken'], ensemble_results['time_taken']]
    })

    print(results_df)

if __name__ == "__main__":
    train_and_save_models(X_train, X_test, y_train, y_test)