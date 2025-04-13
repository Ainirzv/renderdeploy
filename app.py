from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample data
texts = [
    "Free iPhone! Click here to win now",
    "Hackathon starts tomorrow at 10am",
    "Assignment deadline extended to Monday",
    "I lost my wallet near the cafeteria",
    "Found: AirPods in Lecture Hall 3"
]
labels = ['spam', 'events', 'academics', 'lost', 'found']

# Train model
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(X_train, y_train)

# Flask app
app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return "<h1>aini's text classifier</h1>"

@app.route("/classify", methods=['POST'])
def classify():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide text in JSON body like {\"text\": \"your message\"}'}), 400

    input_text = data['text']
    prediction = model.predict([input_text])[0]
    return jsonify({'text': input_text, 'category': prediction})

if __name__ == "__main__":
    app.run(debug=True)
