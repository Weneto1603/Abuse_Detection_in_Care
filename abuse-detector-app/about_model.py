import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="About the prediction model",
    page_icon="🚨"
)

st.title("🤖 About the App")
st.write("""Welcome to the Abuse Detection App! This tool is designed to help identify abusive language in care settings, 
providing support to caregivers and help create a safe and respectful environment for all.""")
st.subheader("The Training Data")
st.markdown("""
    The models used in this app were trained on a unique dataset specifically developed for detecting abusive language 
    in care environments. The dataset was compiled from various care home documentaries, capturing real-life interactions.
    The dataset consists of **566** rows of data. Each row represents a single spoken utterance from the documentaries
    , providing a diverse range of language used in care settings. The dataset includes a mixture of both abusive and 
    non-abusive language, allowing the models to learn and differentiate between different types of utterances.
""")

st.subheader("The Prediction Models")
st.markdown("You can choose from two models: **Gradient Boosting Classifier** and **DistilBERT**.")
st.markdown("The Gradient Boosting Classifier offers better interpretability and explainability, suitable for users who"
            "prioritize understanding the decision-making process of the model.")
st.markdown("The DistilBERT model offers superior performance, making it more suitable for users who prioritise the "
            "accuracy of the predictions.")

st.subheader("Model Performance Metrics")

df = pd.DataFrame(
    [
        {"Model": "DistilBERT", "Accuracy": 0.9412, "Precision": 0.9426, "Recall": 0.9412, "F1": 0.9406, "ROC-AUC": 0.9298},
        {"Model": "Gradient Boosting Classifier", "Accuracy": 0.9176, "Precision": 0.9274, "Recall": 0.9176, "F1": 0.9154, "ROC-AUC": 0.8939}
    ]
)

st.dataframe(df, use_container_width=True)

st.subheader("Confusion Matrices")

st.image('images/distilbert_matrix.png', width=533)
st.image('images/gbc_matrix.png', width=533)

