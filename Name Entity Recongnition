import random
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Step to download necessary NLTK resources
nltk.download('vader_lexicon')

# Step 1: Generate a large text corpus (simulated)
sample_sentences = [
    "I love natural language processing.",
    "This is a fantastic tool for sentiment analysis.",
    "I dislike when my model doesn't perform well.",
    "Machine learning is very interesting.",
    "The weather today is terrible, but I feel good.",
    "I am excited about learning new things!",
    "This is the worst movie I have ever seen.",
    "The food was great and the service was amazing!",
    "I don't think this will work as expected.",
    "I had a wonderful experience with this product."
]

# Create a larger corpus by repeating the sample sentences
corpus = " ".join(sample_sentences * 20000)  # Making it larger

# Split the corpus into sentences for analysis
sentences = corpus.split('. ')  # Simple sentence splitting

# Step 2: Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()
vader_results = []

for sentence in sentences:
    scores = sia.polarity_scores(sentence)
    vader_results.append((sentence, scores))

# Step 3: Sentiment Analysis using TextBlob
textblob_results = []

for sentence in sentences:
    blob = TextBlob(sentence)
    textblob_results.append((sentence, blob.sentiment))

# Display the results for the first few sentences
print("VADER Sentiment Analysis Results:")
for i in range(3):  # Display first 3 results
    print(f"Sentence: {vader_results[i][0]}")
    print(f"Scores: {vader_results[i][1]}\n")

print("TextBlob Sentiment Analysis Results:")
for i in range(3):  # Display first 3 results
    print(f"Sentence: {textblob_results[i][0]}")
    print(f"Polarity: {textblob_results[i][1].polarity}, Subjectivity: {textblob_results[i][1].subjectivity}\n")
