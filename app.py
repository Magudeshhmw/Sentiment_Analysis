import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data
try:
    nltk.data.find('corpora/movie_reviews')
except LookupError:
    nltk.download('movie_reviews')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

# Load movie reviews dataset
def load_data():
    documents = []
    labels = []
    
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append(movie_reviews.raw(fileid))
            labels.append(1 if category == 'pos' else 0)
    
    return documents, labels

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# Train Model
def train_model(documents, labels):
    processed_docs = [preprocess_text(doc) for doc in documents]
    
    # Improved Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed_docs)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use LinearSVC for better accuracy, wrapped in CalibratedClassifierCV for probabilities
    svm = LinearSVC(random_state=42, dual='auto')
    classifier = CalibratedClassifierCV(svm) 
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return classifier, vectorizer, accuracy

# VADER sentiment analysis
def vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return "Positive", compound_score
    elif compound_score <= -0.05:
        return "Negative", compound_score
    else:
        return "Neutral", compound_score

# Initialize model and vectorizer
print("Training model...")
documents, labels = load_data()
classifier, vectorizer, accuracy = train_model(documents, labels)
print(f"Model trained with accuracy: {accuracy:.2f}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    user_input = data.get('text', '')
    
    # ML Model prediction
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    
    # Get probabilities
    proba = classifier.predict_proba(input_vector)[0]
    neg_proba = proba[0]
    pos_proba = proba[1]
    
    prediction = classifier.predict(input_vector)[0]
    ml_sentiment = "Positive" if prediction == 1 else "Negative"
    ml_confidence = pos_proba if prediction == 1 else neg_proba
    
    # VADER prediction
    vader_sentiment_label, vader_score = vader_sentiment(user_input)
    
    # Determine final sentiment
    # If input is very short, trust VADER more. Otherwise trust ML model if confidence is high.
    if len(user_input.split()) < 5:
        final_sentiment = vader_sentiment_label
    else:
        final_sentiment = ml_sentiment
    
    return jsonify({
        'input': user_input,
        'ml_sentiment': ml_sentiment,
        'ml_confidence': float(ml_confidence),
        'ml_pos_proba': float(pos_proba),
        'ml_neg_proba': float(neg_proba),
        'vader': vader_sentiment_label,
        'vader_score': vader_score,
        'final': final_sentiment,
        'accuracy': accuracy
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

