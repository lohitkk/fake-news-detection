# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import string

# Step 1: Load the data
true_news_df = pd.read_csv('/content/True (1).csv')
fake_news_df = pd.read_csv('/content/Fake (1).csv')

# Step 2: Preprocess the data
def clean_text(text):
    # Remove punctuation, lowercase text, and remove numbers
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text.lower())
    text = re.sub(r'\d+', '', text)
    return text

# Apply text cleaning to both datasets
true_news_df['text'] = true_news_df['text'].apply(clean_text)
fake_news_df['text'] = fake_news_df['text'].apply(clean_text)

# Add a target column to distinguish between real and fake news
true_news_df['label'] = 1  # Real news
fake_news_df['label'] = 0  # Fake news

# Combine both datasets
data = pd.concat([true_news_df, fake_news_df], ignore_index=True)

# Step 3: Split the data into training and testing sets
X = data['text']
y = data['label']

# Check for missing values
X = X.fillna('')
y = y.fillna(0)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Evaluate Random Forest model
y_pred_rf = rf_model.predict(X_test_tfidf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%')
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Fake', 'Real']))

# Step 6: Train an XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_tfidf, y_train)

# Evaluate XGBoost model
y_pred_xgb = xgb_model.predict(X_test_tfidf)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'XGBoost Model Accuracy: {accuracy_xgb * 100:.2f}%')
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Fake', 'Real']))

# Step 7: Prediction function for user input
def predict_news(model, news):
    news_cleaned = clean_text(news)
    news_tfidf = vectorizer.transform([news_cleaned])
    prediction = model.predict(news_tfidf)[0]
    if prediction == 1:
        print("The news is REAL.")
    else:
        print("The news is FAKE.")

# Example usage
user_news = input("Enter the news text: ")
print("\nPrediction with Random Forest Model:")
predict_news(rf_model, user_news)
print("\nPrediction with XGBoost Model:")
predict_news(xgb_model, user_news)