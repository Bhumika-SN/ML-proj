from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)
# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    return text

# API route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = clean_text(data["text"])

    vector = vectorizer.transform([text])
    result = model.predict(vector)[0]

    return jsonify({
        "result": "FAKE 🚨" if result == 1 else "REAL ✅"
    })

# Run server
if __name__ == "__main__":
    app.run(debug=True)