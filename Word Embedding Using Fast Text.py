import random
from gensim.utils import simple_preprocess
from gensim.models import FastText

# Step 1: Simulating a text corpus with random sentences
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
corpus = sample_sentences * 1000  # Making it larger

# Step 2: Tokenize the corpus
tokenized_corpus = [simple_preprocess(sentence) for sentence in corpus]

# Step 3: Train FastText model
model = FastText(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=1)

# Step 4: Explore the word embeddings
# Get vector for a specific word
word_vector = model.wv['natural']
print(f"Vector for 'natural': {word_vector}")

# Find most similar words
similar_words = model.wv.most_similar('natural', topn=5)
print("Most similar words to 'natural':")
for word, similarity in similar_words:
    print(f"{word}: {similarity}")

# Step 5: Save the model (optional)
model.save("fasttext_model.bin")

# Step 6: Load the model (optional)
# loaded_model = FastText.load("fasttext_model.bin")
