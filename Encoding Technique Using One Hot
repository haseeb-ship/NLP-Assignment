import numpy as np
from sklearn.preprocessing import OneHotEncoder

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

# Step 2: Tokenization
words = corpus.split()
unique_words = set(words)

# Step 3: One-Hot Encoding
# Reshape the unique words for One-Hot Encoder
unique_words = np.array(list(unique_words)).reshape(-1, 1)

# Create OneHotEncoder (updated parameter)
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform the unique words
one_hot_encoded = encoder.fit_transform(unique_words)

# Example: Show the One-Hot encoding for some words
for i in range(5):  # Display One-Hot encoding for the first 5 unique words
    print(f"Word: {unique_words[i][0]}, One-Hot: {one_hot_encoded[i]}")
