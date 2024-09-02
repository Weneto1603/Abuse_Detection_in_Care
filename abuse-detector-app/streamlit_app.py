import streamlit as st

text_predict = st.Page("text_predict.py", title="📝  Text input")
audio_predict = st.Page("audio_predict.py", title="🔊  Upload and audio file")
mic_predict = st.Page("mic_predict.py", title="🎤️ Use your microphone")
about_model = st.Page("about_model.py", title="🤖  About the app")

pg = st.navigation([text_predict, audio_predict, mic_predict, about_model])
pg.run()