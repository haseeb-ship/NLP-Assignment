import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag



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
sentences = sent_tokenize(corpus)

# Step 3: POS Tagging
tagged_sentences = [pos_tag(word_tokenize(sentence)) for sentence in sentences]

# Display the POS tags for the first few sentences
for i in range(3):  # Displaying the first 3 sentences with their POS tags
    print(f"Sentence {i + 1}: {sentences[i]}")
    print(f"POS Tags: {tagged_sentences[i]}\n")
