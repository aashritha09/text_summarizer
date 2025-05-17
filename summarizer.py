import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq

# Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')

# Sample input text
text = """Text summarization is the process of shortening a text while preserving its meaning. 
It helps reduce reading time, makes large documents easier to understand, and extracts important points.
There are two types of summarization: extractive and abstractive. This tool uses extractive summarization."""

# Step 1: Tokenize the text and remove stopwords
stop_words = set(stopwords.words("english"))
words = word_tokenize(text)

# Frequency table
freq_table = {}
for word in words:
    word = word.lower()
    if word not in stop_words and word.isalnum():
        freq_table[word] = freq_table.get(word, 0) + 1

# Step 2: Score sentences based on word frequencies
sentences = sent_tokenize(text)
sentence_scores = {}
for sent in sentences:
    for word in word_tokenize(sent.lower()):
        if word in freq_table:
            sentence_scores[sent] = sentence_scores.get(sent, 0) + freq_table[word]

# Step 3: Get the top 2 sentences
summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)
summary = " ".join(summary_sentences)

# Output
print("Original Text:\n", text)
print("\nSummary:\n", summary)