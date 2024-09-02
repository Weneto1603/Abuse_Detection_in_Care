import streamlit as st
from utils import *

st.set_page_config(
    page_title="Abuse detection - Audio",
    page_icon="🚨"
)

st.title("🔉 Care Home Abuse Detection")
st.write('')
st.markdown('##### Upload an audio file to check if it contains any abusive language.')

# 11/08 added option for users to choose the prediction model
model_option = st.selectbox('Choose a model:', ('DistilBERT', 'Gradient Boosting Classifier'))

# prompt the user to upload an audio file
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if st.button('Detect Abuse'):
    if audio_file:
        with st.spinner('Converting audio to text...'):
            text = convert_audio_to_text(audio_file)
            # added error handling to account for when the audio file has too much noise and when there is a connection error
            if text == "Google Speech Recognition could not understand the audio":
                st.image('images/sad.png', width=233)
                st.write("Sorry, I cannot recognise the speech in the audio file. Please upload a clearer audio file.")
            elif text == "Could not request results from Google Speech Recognition service":
                st.image('images/sad.png', width=233)
                st.write("Sorry, the audio-to-text function is not available at the minute. Please try again later.")
            else:
                st.write("Transcribed text:")
                st.write(text)

                sentences = sent_tokenize(text)
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
                    st.write('##### No abusive language detected.')
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
    else:
        st.write('Please upload an audio file.')