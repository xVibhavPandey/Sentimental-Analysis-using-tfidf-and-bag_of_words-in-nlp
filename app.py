import string
import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


st.set_page_config(page_title="Emotion Detection App", page_icon="🎭", layout="centered")


@st.cache_resource
def load_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

stop_words = load_nltk()


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('logistic_regression_model.pkl')
    labels = joblib.load('label_map.pkl')
    return vectorizer, model, labels

vectorizer, model, label_map = load_artifacts()


def clean_text(text: str) -> str:
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Remove digits
    text = ''.join([ch for ch in text if not ch.isdigit()])
    # 4. Remove non-ascii (emojis, symbols)
    text = ''.join([ch for ch in text if ch.isascii()])
    # 5. Tokenize and filter stopwords
    words = word_tokenize(text)
    cleaned_words = [w for w in words if w not in stop_words]
    return ' '.join(cleaned_words)


EMOTION_META = {
    "sadness": {"emoji": "😢", "color": "#4A90E2"},
    "anger": {"emoji": "😠", "color": "#E74C3C"},
    "love": {"emoji": "❤️", "color": "#E91E63"},
    "surprise": {"emoji": "😲", "color": "#F39C12"},
    "fear": {"emoji": "😨", "color": "#8E44AD"},
    "joy": {"emoji": "😄", "color": "#2ECC71"}
}


st.title("🎭 Emotion Classifier")
st.write("Predict emotional sentiment using **TF-IDF + Logistic Regression**.")

user_text = st.text_area(
    "Enter a sentence to analyze:",
    placeholder="e.g., I am ever feeling nostalgic about the fireplace...",
    height=120
)

if st.button("Analyze Emotion", type="primary"):
    if not user_text.strip():
        st.warning("Please provide some text to analyze.")
    else:
        # Preprocess
        processed_text = clean_text(user_text)

        if not processed_text.strip():
            st.warning("The input contains only stopwords/symbols. Please enter a fuller sentence.")
        else:
            # Transform and predict
            vec_input = vectorizer.transform([processed_text])
            pred_code = model.predict(vec_input)[0]
            pred_label = label_map[pred_code]
            pred_probs = model.predict_proba(vec_input)[0]

            meta = EMOTION_META.get(pred_label, {"emoji": "✨", "color": "#333333"})

            # Display prediction banner
            st.markdown(
                f"""
                <div style="background-color: {meta['color']}15; border-left: 5px solid {meta['color']}; padding: 16px; border-radius: 6px; margin: 15px 0;">
                    <h3 style="margin: 0; color: {meta['color']};">{meta['emoji']} Detected Emotion: {pred_label.upper()}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display probability breakdown
            st.subheader("Confidence Scores")
            prob_df = pd.DataFrame({
                "Emotion": [label_map[i] for i in range(len(label_map))],
                "Probability": pred_probs
            }).sort_values(by="Probability", ascending=False).reset_index(drop=True)

            st.bar_chart(prob_df.set_index("Emotion"))

            with st.expander("View Preprocessed Text"):
                st.code(processed_text, language="text")