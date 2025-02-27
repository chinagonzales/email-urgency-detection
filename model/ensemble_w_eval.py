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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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

# --- Enhanced Multinomial Naive Bayes --- #
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
            if self.selector is None:
                raise ValueError("Feature selector has not been initialized. Ensure fit() is called before transform().")
            X_selected = self.selector.transform(X_combined)
        else:
            X_selected = X_combined

        return X_selected

    def predict(self, X):
        if self.feature_log_prob is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")

        X_selected = self.transform(X)
        
        if X_selected.shape[1] != self.feature_log_prob.shape[1]:
            raise ValueError(
                f"Feature mismatch: Model expects {self.feature_log_prob.shape[1]} features, "
                f"but got {X_selected.shape[1]}. Ensure consistent feature selection."
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

# Resampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
print("Class Distribution After Resampling:")
print(pd.Series(y_resampled).value_counts())
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

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

def perform_cross_validation(model, X_train, y_train, model_name, cv_folds=5):
    print(f"Performing {cv_folds}-fold Cross-Validation for {model_name}...")
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores for {model_name}: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f} | Standard Deviation: {scores.std():.4f}\n")


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

def train_and_save_models():
     # --- Fit EnhancedMNB --- #
    print("Training EnhancedMNB...")
    mnb_model = EnhancedMNB(ngram_range=(1, 2))
    mnb_model.fit(X_train, y_train)  

    # Perform Cross-Validation for EnhancedMNB
    perform_cross_validation(mnb_model, X_train, y_train, "Enhanced MNB")
    
    # Evaluate MNB before hyperparameter tuning
    mnb_results = evaluate_model(mnb_model, X_test, y_test, "Enhanced MNB")

    # --- EnhancedMNB with Grid Search --- #
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0],
        'ngram_range': [(1, 1), (1, 2)],
        'k_best': [100, 500, 1000]
    }

    grid_search = GridSearchCV(mnb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found for EnhancedMNB:", grid_search.best_params_)
    best_mnb_model = grid_search.best_estimator_

    # Perform Cross-Validation for EnhancedMNB
    perform_cross_validation(best_mnb_model, X_train, y_train, "Enhanced MNB (with hyperparameter tuning)")

    # Evaluate MNB after hyperparameter tuning
    mnb_results = evaluate_model(best_mnb_model, X_test, y_test, "Enhanced MNB (with hyperparameter tuning)")

    # Plot Confusion Matrix for EnhancedMNB
    y_pred_mnb = best_mnb_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_mnb, "Enhanced MNB (with hyperparameter tuning)")

    # --- Logistic Regression with Bag of Words --- #
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', CountVectorizer(ngram_range=(1, 2)), 'processed_text')
        ]
    )

    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('log_reg', LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=100))
    ])

    print("Training Logistic Regression Model...")
    lr_pipeline.fit(X_train, y_train)

    # Perform Cross-Validation for Logistic Regression
    perform_cross_validation(lr_pipeline, X_train, y_train, "Logistic Regression")

    # Evaluate LR
    lr_results = evaluate_model(lr_pipeline, X_test, y_test, "Enhanced MNB (with hyperparameter tuning)")


    # Plot Confusion Matrix for Logistic Regression
    y_pred_lr = lr_pipeline.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")

    # --- Ensemble Model --- #
    ensemble = VotingClassifier(estimators=[
        ('enhanced_mnb', best_mnb_model),
        ('lr_pipeline', lr_pipeline)
    ], voting='soft')

    # Perform Cross-Validation for Ensemble
    perform_cross_validation(ensemble, X_train, y_train, "Ensemble Model")

    print("Training Ensemble Model...")
    ensemble.fit(X_train, y_train)

    #Evaluate Ensemble
    ensemble_results = evaluate_model(ensemble, X_test, y_test, "Ensemble Model")

    # Plot Confusion Matrix for Ensemble Model
    y_pred_ensemble = ensemble.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_ensemble, "Ensemble Model")

    # Save models
    joblib.dump(ensemble, 'ensemble_model.pkl')
    print("Model saved successfully!")

    # --- Model Evaluation --- #
    mnb_results = evaluate_model(best_mnb_model, X_test, y_test, "Enhanced MNB")
    lr_results = evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression")
    ensemble_results = evaluate_model(ensemble, X_test, y_test, "Ensemble Model")

    results_df = pd.DataFrame({
        'Model': ['Enhanced MNB', 'Logistic Regression', 'Ensemble'],
        'Accuracy': [mnb_results['accuracy'], lr_results['accuracy'], ensemble_results['accuracy']],
        'Precision': [mnb_results['precision'], lr_results['precision'], ensemble_results['precision']],
        'Time Taken': [mnb_results['time_taken'], lr_results['time_taken'], ensemble_results['time_taken']]
    })

    print(results_df)


if __name__ == "__main__":
    train_and_save_models()