import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

# Download required NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('punkt_tab')
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

# Preprocess text for Naive Bayes
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# Train Naive Bayes model
def train_model(documents, labels):
    processed_docs = [preprocess_text(doc) for doc in documents]
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(processed_docs)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    classifier = MultinomialNB()
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

# Main function
def main():
    # Load and train model
    documents, labels = load_data()
    classifier, vectorizer, accuracy = train_model(documents, labels)
    
    print(f"Naive Bayes Model Accuracy on Movie Reviews Dataset: {accuracy:.2f}")
    
    # Get user input
    user_input = input("Enter your text (or press Enter to use default 'I like this code very much and working correctly'): ")
    if not user_input.strip():
        user_input = "I like this code very much and working correctly"
    
    # Naive Bayes prediction
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    nb_prediction = classifier.predict(input_vector)
    nb_sentiment = "Positive" if nb_prediction[0] == 1 else "Negative"
    nb_confidence = classifier.predict_proba(input_vector)[0][nb_prediction[0]]
    
    # VADER prediction
    vader_sentiment_label, vader_score = vader_sentiment(user_input)
    
    # Determine final sentiment
    final_sentiment = vader_sentiment_label if len(user_input.split()) < 10 or nb_confidence < 0.7 else nb_sentiment
    
    # Print results
    print("\nResults:")
    print(f"Input Text: {user_input}")
    print(f"Naive Bayes Prediction: {nb_sentiment} (Confidence: {nb_confidence:.2f})")
    print(f"VADER Prediction: {vader_sentiment_label} (Compound Score: {vader_score:.2f})")
    print("\nFinal Output Result:")
    print(final_sentiment.upper())
    print("Additional Sentiments:")
    print(f"- Negative: {'Yes' if final_sentiment == 'Negative' else 'No'}")
    print(f"- Neutral: {'Yes' if final_sentiment == 'Neutral' else 'No'}")
    print(f"- Good: {'Yes' if final_sentiment == 'Positive' else 'No'}")
    
    if len(user_input.split()) < 10 or nb_confidence < 0.7:
        print("\nNote: The input is short or the Naive Bayes model has low confidence. VADER's prediction was used as the final result.")

if __name__ == "__main__":
    main()