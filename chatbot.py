import json
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (only the first time you run it)
nltk.download('punkt')
nltk.download('wordnet')

# Load FAQs from file
with open('faqs.json', 'r') as f:
    faqs = json.load(f)

questions = [faq['question'] for faq in faqs]
answers = [faq['answer'] for faq in faqs]

# Text preprocessing
lemmatizer = nltk.WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

preprocessed_questions = [preprocess(q) for q in questions]

# Vectorize questions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

def chatbot_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    idx = similarity.argmax()
    return answers[idx]

# Chat loop
print("FAQ Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
