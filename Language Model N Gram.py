import random
from collections import Counter
from nltk import ngrams
import nltk

# Download necessary NLTK resources
nltk.download('punkt')

# Step 1: Generate a large text corpus (simulated)
sample_sentences = [
    "Natural language processing is a fascinating field.",
    "Word embeddings help in understanding the meaning of words.",
    "FastText is an extension of Word2Vec.",
    "Machine learning techniques are widely used in NLP.",
    "Text classification involves categorizing text into predefined classes.",
    "Deep learning has significantly improved the state of NLP.",
    "Sentiment analysis is used to determine the sentiment of text.",
    "Named entity recognition identifies and classifies key entities in text.",
    "Chatbots are an application of NLP that interacts with users.",
    "Tokenization is the process of splitting text into individual words."
]

# Create a larger corpus by repeating the sample sentences
corpus = " ".join(sample_sentences * 20000)  # Making it larger

# Step 2: Preprocess the text
# Tokenization
tokens = nltk.word_tokenize(corpus)

# Step 3: Generate N-grams
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Example with 2-grams (bigrams) and 3-grams
bigrams = generate_ngrams(tokens, 2)
trigrams = generate_ngrams(tokens, 3)

# Step 4: Count frequencies of each N-gram
bigram_counts = Counter(bigrams)
trigram_counts = Counter(trigrams)

# Display some results
print("Top 5 Bigrams:")
for bigram, count in bigram_counts.most_common(5):
    print(f"{bigram}: {count}")

print("\nTop 5 Trigrams:")
for trigram, count in trigram_counts.most_common(5):
    print(f"{trigram}: {count}")
