import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from nltk.corpus import stopwords
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# Load Dataset
def load_dataset(file_path):
    """Load the Q&A dataset from a CSV file."""
    return pd.read_csv(file_path)

# Remove Stop Words
def remove_stop_words(text):
    """
    Remove stop words from a given text.
    """
    words = str(text).split()
    filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
    return ' '.join(filtered_words)

# Word Correction Function
def correct_word(word, vocabulary):
    """
    Correct a misspelled word using the closest match in the vocabulary.
    """
    closest_matches = get_close_matches(word, vocabulary, n=1, cutoff=0.7)  # Adjust cutoff for sensitivity
    return closest_matches[0] if closest_matches else word

# Correct Grammar in a Query
def correct_query(query, vocabulary):
    """
    Correct spelling errors in the query by checking each word.
    """
    # Remove stop words from the query
    query = remove_stop_words(query)
    words = query.split()
    corrected_words = [correct_word(word, vocabulary) for word in words]
    return ' '.join(corrected_words)

# Vectorize Dataset Questions
def vectorize_questions(questions):
    """
    Generate TF-IDF vectors for all questions in the dataset.
    """
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors

# Find Most Similar Question
def find_most_similar_question(query, vectorizer, question_vectors, questions):
    """
    Find the most similar question in the dataset using cosine similarity.
    """
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, question_vectors)
    most_similar_idx = np.argmax(similarities)
    return questions[most_similar_idx]

# Chatbot Response
def chatbot_response(user_query, vocabulary, vectorizer, question_vectors, dataset):
    """
    Generate a chatbot response based on the user query.
    """
    # Correct the query for spelling errors and remove stop words
    corrected_query = correct_query(user_query, vocabulary)

    # Find the most similar question
    dataset_questions = dataset['Question'].tolist()
    most_similar_question = find_most_similar_question(corrected_query, vectorizer, question_vectors, dataset_questions)
    
    # Retrieve the corresponding answer
    response = dataset[dataset['Question'] == most_similar_question]['Answer'].values[0]
    return response
