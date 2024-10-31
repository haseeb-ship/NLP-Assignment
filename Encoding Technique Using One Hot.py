import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Step 1: Generate a large text corpus (simulated)
sample_sentences = [
    """The quick brown fox jumps over the lazy dog. It is a common pangram.
Natural Language Processing (NLP) is a subfield of artificial intelligence.
Machine learning algorithms can be supervised or unsupervised.
Text mining and analysis are critical in understanding large datasets.
Many researchers focus on deep learning techniques for improved accuracy.
A journey of a thousand miles begins with a single step.
The rain in Spain stays mainly in the plain.
All work and no play makes Jack a dull boy.
An apple a day keeps the doctor away.
The early bird catches the worm, or so they say.
A picture is worth a thousand words, they often claim.
Better late than never, or so the saying goes.
A watched pot never boils, but it does simmer.
Actions speak louder than words, as the old adage suggests.
A penny saved is a penny earned, they say wisely.
Time flies when you're having fun, or so it feels.
The grass is always greener on the other side.
You can’t judge a book by its cover, or so they warn.
When in Rome, do as the Romans do.
Haste makes waste, they say when rushed.
Out of sight, out of mind, a common refrain.
The pen is mightier than the sword, they claim.
A friend in need is a friend indeed, or so they believe.
Laughter is the best medicine, they often assert.
Rome wasn’t built in a day, as the saying goes.
Every cloud has a silver lining, they say hopefully.
Don’t put all your eggs in one basket, they advise.
The squeaky wheel gets the grease, as they often note.
The best things in life are free, or so they believe.
You can’t have your cake and eat it too, they warn.
The best laid plans of mice and men often go awry.
A bird in the hand is worth two in the bush.
Curiosity killed the cat, but satisfaction brought it back.
The devil is in the details, they often remind us.
All good things must come to an end, they say.
If it ain’t broke, don’t fix it, as the saying goes.
To err is human, to forgive divine.
Fortune favors the bold, they claim.
No pain, no gain, a common motto for success.
Blood is thicker than water, or so they say.
Every rose has its thorn, a bittersweet truth.
A chain is only as strong as its weakest link.
A journey well begun is half done, or so they suggest.
There’s no place like home, they often feel.
You can lead a horse to water, but you can’t make it drink.
A rolling stone gathers no moss, they remind us.
The more things change, the more they stay the same.
Hope for the best, but prepare for the worst.
Time heals all wounds, they say with time.
Where there’s smoke, there’s fire, as they warn.
Don’t count your chickens before they hatch.
Great minds think alike, they often remark.
What goes around comes around, they say wisely.
All that glitters is not gold, a reminder of reality."""
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
