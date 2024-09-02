import streamlit as st
from utils import convert_mic_audio_to_text, predict_abuse, sent_tokenize
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

plt.rcParams.update({'font.size': 5})

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_bytes = b""

    def recv(self, frame):
        self.audio_bytes = frame.to_ndarray().tobytes()

st.set_page_config(
    page_title="Abuse detection - Text",
    page_icon="🚨"
)


st.title("🎤️Care Home Abuse Prediction")
st.write("Use your microphone to check if any speech contains abusive language.")

ctx = webrtc_streamer(key="audio", mode=WebRtcMode.SENDONLY, audio_processor_factory=AudioProcessor)
if ctx.audio_processor and ctx.audio_processor.audio_bytes:
    audio_bytes = ctx.audio_processor.audio_bytes
    st.audio(audio_bytes, format="audio/wav")

    if st.button('Detect Abuse'):
        with st.spinner('Converting audio to text...'):
            text = convert_mic_audio_to_text(audio_bytes)
            st.write("Transcribed text:")
            st.write(text)

            sentences = sent_tokenize(text)
            predictions, prediction_probabilities = predict_abuse(sentences)
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
                fig, ax = plt.subplots(figsize=(3, 2))
                ax.bar(['Abusive', 'Non-abusive'], [probability, 1 - probability], color=['red', 'blue'])
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probability')
                st.pyplot(fig)