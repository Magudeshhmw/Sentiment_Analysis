import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon')

# VADER sentiment analysis
def analyze_sentiment(text):
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
    # Default input
    default_input = "I like this code very much and working correctly"
    
    # Analyze sentiment
    sentiment, score = analyze_sentiment(default_input)
    
    # Print results
    print(f"Input Text: {default_input}")
    print(f"Final Output Result: {sentiment.upper()}")
    print(f"Compound Score: {score:.2f}")
    print("Additional Sentiments:")
    print(f"- Negative: {'Yes' if sentiment == 'Negative' else 'No'}")
    print(f"- Neutral: {'Yes' if sentiment == 'Neutral' else 'No'}")
    print(f"- Good: {'Yes' if sentiment == 'Positive' else 'No'}")

if __name__ == "__main__":
    main()