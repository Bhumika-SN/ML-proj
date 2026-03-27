import streamlit as st
import pandas as pd
import re
import nltk
import requests
import matplotlib.pyplot as plt
import seaborn as sns

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Download stopwords
nltk.download('stopwords')

# ---------------- DATA ----------------
stop_words = set(stopwords.words('english'))

data = pd.read_csv("fake_job_postings.csv")  # your dataset
data = data.fillna("")

# Combine text columns
data["text"] = (
    data["title"] + " " +
    data["company_profile"] + " " +
    data["description"] + " " +
    data["requirements"]
)

# Target
y = data["fraudulent"]

# ---------------- CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data["text"] = data["text"].apply(clean_text)

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data["text"])

# ---------------- TRAIN-TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ---------------- URL EXTRACTION ----------------
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text
    except:
        return ""

# ---------------- PREDICTION ----------------
def predict_job(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)
    proba = model.predict_proba(vector)[0]
    confidence = max(proba)

    label = "FAKE 🚨" if result[0] == 1 else "REAL ✅"
    return f"{label} (Confidence: {confidence:.2f})"

# ---------------- UI ----------------

st.title("🧠 Fake Job Detection System")

st.subheader(f"📊 Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ---------------- TEXT INPUT ----------------

st.write("Enter job description:")

user_input = st.text_area("Job Description")

if st.button("Predict from Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict_job(user_input)

        if "FAKE" in result:
            st.error(f"🚨 {result}")
        else:
            st.success(f"✅ {result}")

# ---------------- URL INPUT ----------------

st.write("Or enter job link:")

url_input = st.text_input("Job URL")

if st.button("Predict from URL"):
    if url_input.strip() == "":
        st.warning("Please enter a URL")
    else:
        extracted_text = extract_text_from_url(url_input)

        if extracted_text == "":
            st.error("Could not extract data from URL")
        else:
            result = predict_job(extracted_text)

            if "FAKE" in result:
                st.error(f"🚨 {result}")
            else:
                st.success(f"✅ {result}")