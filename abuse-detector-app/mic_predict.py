import speech_recognition as sr
from utils import *
import matplotlib.pyplot as plt
import streamlit as st

# function to recognise speech from mic input
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        with st.spinner("Listening...."):
            audio = recognizer.listen(source)
            response = {
                "success": True,
                "error": None,
                "transcription": None
            }
            try:
                response["transcription"] = recognizer.recognize_whisper(audio)
            except sr.RequestError:
                # if the speech recognition API is not available
                response["success"] = False
                response["error"] = "API unavailable"
            except sr.UnknownValueError:
                # if speech was unintelligible
                response["error"] = "Unable to recognize speech"
        return response


plt.rcParams.update({'font.size': 5})

st.set_page_config(
    page_title="Abuse detection - Microphone",
    page_icon="🚨"
)

st.title("🎤️Care Home Abuse Prediction")
st.write("Use your microphone to check if any speech contains abusive language.")

# 11/08 added option for users to choose the prediction model
model_option = st.selectbox('Choose a model:', ('DistilBERT', 'Gradient Boosting Classifier'))

recognizer = sr.Recognizer()
microphone = sr.Microphone()

recording = st.button("Start Recording")

if recording:
    response = recognize_speech_from_mic(recognizer, microphone)
    # if speech is recognised, print the transcribed text
    if response["transcription"]:
        st.write("Transcribed text:")
        st.write(response["transcription"])
        #
        # # added error handling if unable to transcribe audio input
        # if response["error"] == "API unavailable":
        #     st.write("Sorry, this service is unavailable at the moment. Please try again later.")
        # elif response["error"] == "Unable to recognize speech":
        #     st.write("Sorry, I didn't catch that. Please try again.")

        sentences = sent_tokenize(response["transcription"])
        if model_option == 'Gradient Boosting Classifier':
            predictions, prediction_probabilities = predict_abuse(sentences)
        elif model_option == 'DistilBERT':
            predictions, prediction_probabilities = predict_abuse_bert(sentences)
        abusive_sentences = [(sentences[i], prediction_probabilities[i][1]) for i in range(len(predictions)) if predictions[i] == 1]
        all_sentences = [(sentences[i], prediction_probabilities[i][1]) for i in range(len(predictions))]

        if abusive_sentences:
            st.image('images/warning.png', width=233)
            st.write('')
            st.markdown('##### The audio contains possible abusive language.')
            st.write('')
            st.write('Here are the sentences predicted to be abusive:')
            for sentence, probability in abusive_sentences:
                st.write(f'- {sentence} (Probability: {probability:.2f})')
        else:
            st.image('images/tick.png', width=233)
            st.write('')
            st.write('##### No abusive sentences detected.')
        st.divider()
        st.markdown('**Here are the prediction probabilities for all sentences:**')
        for sentence, probability in all_sentences:
            st.write(f'- {sentence}')
            labels = ['Abusive', 'Non-abusive']
            sizes = [probability, 1 - probability]
            colors = ['peachpuff', 'lightsteelblue']
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title('Prediction Probability')
            st.pyplot(fig)
    # print error message if needed
    else:
        st.write("Sorry, I didn't catch that. Please try again.")
        if response["error"]:
            st.write(f"Error: {response['error']}")

if not recording:
    st.write("Click 'Start Recording' to begin capturing audio.")
