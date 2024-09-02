import streamlit as st
from utils import *

# nltk.download('punkt')
# nltk.download('stopwords')

plt.rcParams.update({'font.size': 3})

# """Streamlit app"""

st.set_page_config(
    page_title="Care Home Abuse Detection"
)

with st.sidebar:
    st.markdown("#### Text prediction")
    st.markdown("#### Audio prediction")
    st.markdown("#### About the prediction model")

st.title('Care Home Abuse Detection')
st.markdown('###### Please enter some text to check if any sentences are abusive.')

# get the user input in text
user_input = st.text_area('Text')

if st.button('Detect Abuse'):
    st.write('')
    if user_input:
        sentences = sent_tokenize(user_input)
        predictions, prediction_probabilities = predict_abuse(sentences)
        abusive_sentences = [(sentences[i], prediction_probabilities[i][1]) for i in range(len(predictions)) if
                             predictions[i] == 1]
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
            # plot a graph showing the probability for each sentence
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.bar(['Abusive', 'Non-abusive'], [probability, 1 - probability], color=['red', 'blue'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probability')
            st.pyplot(fig)

    else:
        st.write('Please enter some text.')