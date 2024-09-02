import streamlit as st
from utils import *
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 3})

st.set_page_config(
    page_title="Abuse detection - Text",
    page_icon="🚨"
)

st.title('📝 Care Home Abuse Detection')
st.write('')
st.markdown('##### Please enter some text to check if any sentences are abusive.')

# 11/08 added option for users to choose the prediction model
model_option = st.selectbox('Choose a model:', ('DistilBERT', 'Gradient Boosting Classifier'))

user_input = st.text_area('Text')

if st.button('Detect Abuse'):
    st.write('')
    if user_input:
        sentences = sent_tokenize(user_input)
        if model_option == 'Gradient Boosting Classifier':
            predictions, prediction_probabilities = predict_abuse(sentences)
        elif model_option == 'DistilBERT':
            predictions, prediction_probabilities = predict_abuse_bert(sentences)
        abusive_sentences = [(sentences[i], prediction_probabilities[i][1]) for i in range(len(predictions)) if predictions[i] == 1]
        all_sentences = [(sentences[i], prediction_probabilities[i][1]) for i in range(len(predictions))]

        if abusive_sentences:
            st.image('images/warning.png', width=233)
            st.write('')
            st.markdown('##### The text contains possible abusive language.')
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
    else:
        st.write('Please enter some text.')