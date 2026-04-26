from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///neurolearn.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))

class Resource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    content = db.Column(db.Text)

class StudyHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    topic = db.Column(db.String(200))

with app.app_context():
    db.create_all()

vectorizer = TfidfVectorizer(stop_words='english')
corpus = []

@app.route('/')
def home():
    return "NeuroLearn AI Backend Running"

@app.route('/upload', methods=['POST'])
def upload():
    data = request.json
    title = data.get('title')
    content = data.get('content')

    item = Resource(title=title, content=content)
    db.session.add(item)
    db.session.commit()

    corpus.append(content)
    return jsonify({"message": "Uploaded successfully"})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query = data.get('query')

    if not corpus:
        return jsonify({"error": "No data available"})

    docs = corpus + [query]
    tfidf_matrix = vectorizer.fit_transform(docs)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    return jsonify({"recommendations": similarity[0].tolist()})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    return jsonify({"answer": f"AI Response: {question}"})

@app.route('/youtube', methods=['GET'])
def youtube():
    query = request.args.get('q')
    return jsonify({"videos": [{"title": query, "url": "https://youtube.com"}]})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)