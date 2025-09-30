import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

# اقرأ الداتا
df = pd.read_csv("train.csv")

# Vectorize الأسئلة
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Question"])

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    idx = similarities.argmax()
    response = df["Answer"].iloc[idx]
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)

