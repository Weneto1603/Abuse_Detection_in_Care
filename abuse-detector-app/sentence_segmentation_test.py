import nltk
from nltk import word_tokenize, pos_tag

# Example input
text = 'my name is batman i live in london i want a coffee'

# Tokenize the text
tokens = word_tokenize(text)

# POS tagging
pos_tags = pos_tag(tokens)

# Heuristic-based segmentation
sentences = []
current_sentence = []
for i, (word, pos) in enumerate(pos_tags):
    current_sentence.append(word)
    if pos in ['PRP', 'NNP']:  # Pronoun or Proper Noun
        if i != 0:  # Avoid splitting at the start
            sentences.append(' '.join(current_sentence[:-1]))
            current_sentence = [word]

# Add the last sentence
if current_sentence:
    sentences.append(' '.join(current_sentence))

# Display the segmented sentences
for sentence in sentences:
    print(sentence)

