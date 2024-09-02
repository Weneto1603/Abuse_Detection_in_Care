import joblib
import numpy as np
import nltk
import re
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import speech_recognition as sr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# nltk.download('punkt')
# nltk.download('stopwords')

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# load the trained model (gradient boosting) and the vectorizer from the pkl files
model = joblib.load('model/gradient_boosting_model.pkl')
vectorizer = joblib.load('model/countvectorizer.pkl')

distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-abuse-detection')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-abuse-detection')

plt.rcParams.update({'font.size': 3})


# perform the same normalization as when training the model
def normalize_text(text):
    doc = nlp(text)
    normalized_words = [token.lemma_ for token in doc]
    normalized_text = ' '.join(normalized_words)
    return normalized_text


# perform the same cleaning step
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)  # tokenize
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text


def preprocess_text(text):
    normalized_text = normalize_text(text)
    cleaned_text = clean_text(normalized_text)
    return cleaned_text


# perform the same feature extraction
def extract_features(sentences):
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    text_matrix = vectorizer.transform(preprocessed_sentences)
    pos_features = [' '.join(tag[1] for tag in nltk.pos_tag(word_tokenize(sentence))) for sentence in
                    preprocessed_sentences]
    pos_matrix = vectorizer.transform(pos_features)
    keywords = ['fuck', 'fucking', 'fat', 'bitch', 'cunt', 'dick', 'bastard', 'idiot', 'shut up']
    keyword_matches = np.array(
        [any(keyword in sentence.lower() for keyword in keywords) for sentence in preprocessed_sentences]).reshape(-1,
                                                                                                                   1)
    keyword_matrix = csr_matrix(keyword_matches)
    combined_matrix = hstack([text_matrix, pos_matrix, keyword_matrix])
    return combined_matrix


# predict abuse using the trained model; also using the predict_proba function to get the probability of each prediction
def predict_abuse(sentences):
    features = extract_features(sentences)
    predictions = model.predict(features)
    prediction_probabilities = model.predict_proba(features)
    return predictions, prediction_probabilities

# predict abuse using the distilbert model; as
def predict_abuse_bert(sentences):
    inputs = distilbert_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    outputs = distilbert_model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).numpy()
    probabilities = torch.nn.functional.softmax(logits, dim=1).detach().numpy()
    return predictions, probabilities

def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_whisper(audio)
    except sr.UnknownValueError:
        text = "Whisper Speech Recognition could not understand the audio"
    except sr.RequestError:
        text = f"Could not request results from Whisper Speech Recognition service"
    return text

def convert_mic_audio_to_text(audio_bytes):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioFile(BytesIO(audio_bytes))
    with audio_data as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_whisper(audio)
    except sr.UnknownValueError:
        text = "Whisper Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        text = f"Could not request results from Whisper Speech Recognition service; {e}"
    return text
