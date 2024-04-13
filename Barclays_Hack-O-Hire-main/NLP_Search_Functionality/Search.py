import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "This is a sample document about natural language processing.",
    "NLP helps in processing and understanding text data.",
    "Text mining and NLP techniques are used in information retrieval.",
]

# Tokenize the search query
search_query = "natural language processing techniques"
query_tokens = word_tokenize(search_query.lower())

# Tokenize and vectorize documents
vectorizer = CountVectorizer()
document_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([search_query])

# Calculate cosine similarity between query and documents
similarities = cosine_similarity(query_vector, document_vectors).flatten()

# Rank documents based on similarity
ranked_indices = similarities.argsort()[::-1]
for idx in ranked_indices:
    print(f"Document {idx + 1}: {documents[idx]}")
