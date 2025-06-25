import streamlit as st
import nltk
import speech_recognition as sr
import os
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')

# Prétraitement du texte (nettoyage et normalisation)
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return cleaned

# Chargement du fichier de connaissances (FAQ simple)
def load_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = f.read()
    sentences = nltk.sent_tokenize(corpus)
    return sentences

# Réponse du chatbot en fonction de la similarité
def generate_response(user_input, knowledge_base):
    user_input_processed = preprocess_text(user_input)
    scores = []
    for sentence in knowledge_base:
        sentence_processed = preprocess_text(sentence)
        common_words = set(user_input_processed) & set(sentence_processed)
        scores.append(len(common_words))
    max_score = max(scores)
    if max_score == 0:
        return "Je suis désolé, je ne comprends pas."
    else:
        return knowledge_base[scores.index(max_score)]

# Reconnaissance vocale
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Parlez maintenant...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="fr-FR")
            st.success(f"Vous avez dit : {text}")
            return text
        except sr.UnknownValueError:
            st.error("Désolé, je n'ai pas compris.")
            return ""
        except sr.RequestError:
            st.error("Erreur avec le service de reconnaissance vocale.")
            return ""

# Interface Streamlit
def main():
    st.title("Chatbot Vocal: Conseil Santé")
    st.write("Vous pouvez parler ou écrire une question.")

    # Chargement des données (remplacez le chemin par votre fichier)
    if not os.path.exists("connaissances.txt"):
        st.warning("Veuillez placer un fichier `connaissances.txt` dans le dossier du projet.")
        return
    base_de_connaissance = load_knowledge_base("connaissances.txt")

    # Choix du mode d’entrée
    mode = st.radio("Choisissez un mode d'entrée :", ("Texte", "Voix"))

    if mode == "Texte":
        user_input = st.text_input("Entrez votre question :")
        if st.button("Envoyer") and user_input:
            response = generate_response(user_input, base_de_connaissance)
            st.text_area("Réponse du chatbot :", value=response, height=100)
    else:
        if st.button("Parler"):
            voice_input = recognize_speech()
            if voice_input:
                response = generate_response(voice_input, base_de_connaissance)
                st.text_area("Réponse du chatbot :", value=response, height=100)

if __name__ == "__main__":
    main()
