import streamlit as st
from transformers import pipeline
import torch

# --- Page Setup ---
st.set_page_config(page_title="🧠 NLP Playground", layout="wide")
st.title("🧠 NLP with Hugging Face Transformers")
st.markdown("""
Welcome! This app lets you explore powerful NLP models from Hugging Face 🤗.

Choose a task below, enter your text, and see the results instantly!
""")

# --- Task Selector ---
task = st.selectbox("Select a task", [
    "Sentiment Analysis", "Text Summarization", "Named Entity Recognition"
])

# --- Load Model Based on Task ---
@st.cache_resource
def load_pipeline(task_name):
    if task_name == "Sentiment Analysis":
        return pipeline("sentiment-analysis")
    elif task_name == "Text Summarization":
        return pipeline("summarization")
    elif task_name == "Named Entity Recognition":
        return pipeline("ner", grouped_entities=True)

nlp_pipeline = load_pipeline(task)

# --- User Input ---
text_input = st.text_area("Enter your text here:", height=200)

# --- File Uploader (Bonus) ---
uploaded_file = st.file_uploader("Or upload a .txt file for batch processing", type="txt")

# --- Run Prediction ---
if st.button("Run"):
    if uploaded_file:
        text_input = uploaded_file.read().decode("utf-8")

    if not text_input.strip():
        st.warning("Please enter or upload some text.")
    else:
        st.subheader("🧪 Results")
        with st.spinner("Processing..."):
            result = nlp_pipeline(text_input)

        if task == "Sentiment Analysis":
            for res in result:
                st.write(f"**Label**: {res['label']} | **Confidence**: {res['score']:.2f}")
        elif task == "Text Summarization":
            st.write(result[0]['summary_text'])
        elif task == "Named Entity Recognition":
            for entity in result:
                st.write(f"**{entity['entity_group']}**: {entity['word']} (score: {entity['score']:.2f})")