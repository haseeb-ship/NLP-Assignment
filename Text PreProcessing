import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string

# Sample corpus from Wikipedia (as an example)
texts = [
    "The quick brown fox jumps over the lazy dog. It is a common pangram.",
    "Natural Language Processing (NLP) is a subfield of artificial intelligence.",
    "Machine learning algorithms can be supervised or unsupervised.",
    "Text mining and analysis are critical in understanding large datasets.",
    "Many researchers focus on deep learning techniques for improved accuracy."
]

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Stemming
    stemmer = PorterStemmer()
    tokens_stemmed = [stemmer.stem(token) for token in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens_lemmatized = [lemmatizer.lemmatize(token) for token in tokens_stemmed]

    # Join tokens back to string
    cleaned_text = ' '.join(tokens_lemmatized)
    return cleaned_text

# Preprocess texts and store results
cleaned_texts = [preprocess_text(text) for text in texts]

# Display some examples of before and after
for i in range(len(texts)):  # Show all examples
    print(f"Original Text: {texts[i]}")
    print(f"Cleaned Text: {cleaned_texts[i]}\n")
