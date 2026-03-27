import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("fake_real_job_postings_3000x25.csv")

print("Initial Data:")
print(data.head())

# Handle missing values
data = data.fillna("")

# Combine text columns
data["text"] = (
    data["job_title"] + " " +
    data["job_description"] + " " +
    data["requirements"] + " " +
    data["company_profile"]
)

# Target column
y = data["is_fake"]

# Stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Apply cleaning
data["text"] = data["text"].apply(clean_text)

# Final check
print("\nCleaned Data:")
print(data[["text", "is_fake"]].head())
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["text"])

print("Shape of X:", X.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
def predict_job(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)
    
    return "FAKE" if result[0] == 1 else "REAL"

test_job = "Earn 5000 dollars daily without experience work from home"

print("Prediction:", predict_job(test_job))
print(predict_job("Software engineer required with Python skills"))
print(predict_job("Earn money fast no skills needed"))

print("\n===== Fake Job Detection System =====")

while True:
    user_input = input("\nEnter job description (or type 'exit' to quit):\n")

    if user_input.lower() == "exit":
        print("Exiting... 👋")
        break

    result = predict_job(user_input)
    print("Prediction:", result)