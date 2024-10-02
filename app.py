"""Streamlit Sentiment Analysis App."""

import html
import pickle
import re

import streamlit as st
import tensorflow as tf
from nltk.corpus import stopwords


def set_session() -> None:
    """Set session variables."""

    if "vectorizer" not in st.session_state:
        with open("vectorizer.pickle", "rb") as f:
            st.session_state.vectorizer = pickle.load(f)
    if "model" not in st.session_state:
        st.session_state.model = tf.keras.models.load_model("model.keras")


def preprocess(text: str) -> str:
    """Preprocess text.

    Args:
        text (str): text to be processed.

    Returns:
        str: Processed text.
    """

    text = text.lower()  # lower the text
    text = html.unescape(text)  # parse html entitities
    text = re.sub(
        r"@\S+|https?:\S+|[^A-Za-z0-9]+", " ", text
    ).strip()  # remove the unwanted text
    stop_words = stopwords.words("english")
    tokens = [token for token in text.split() if token not in stop_words]
    text = " ".join(tokens)
    return text


def main() -> None:
    """Main runner."""

    set_session()
    st.header("Sentiment Analyser")
    input_text = st.text_area(label="Input text")
    button = st.button(label="Analyse")

    progress_bar_value = """
    <div style="
            width: 100%;
            height: 30px;
            background-color: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 20px">
        <div style="
            height: 100%;
            width: {}%;
            background-color: #3371ff;
            border-radius: 5px;
            transition: width 2s ease-in-out;">
        </div>
    </div>
    """
    progress_bar = st.markdown(progress_bar_value.format(0), unsafe_allow_html=True)
    percentage = 50
    progress_style = """
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;">
                <span>Negative</span>
                <span>{}</span>
                <span>Positive</span>
            </div>
            """
    progress_text = st.markdown(
        progress_style.format(percentage), unsafe_allow_html=True
    )

    if button and input_text:
        input_text = preprocess(input_text)
        input_vector = st.session_state.vectorizer.transform([input_text]).toarray()
        out = st.session_state.model.predict(input_vector.reshape(1, 1, -1))
        percentage = int(out[0][0] * 100)
        progress_bar.markdown(
            progress_bar_value.format(percentage), unsafe_allow_html=True
        )
        progress_text.markdown(
            progress_style.format(percentage), unsafe_allow_html=True
        )

        # for percentage in range(int(out[0][0] * 100)):
        #     time.sleep(0.01)
        #     progress_bar.markdown(
        #         progress_bar_value.format(percentage), unsafe_allow_html=True
        #     )
        # progress_text.markdown(
        #     progress_style.format(percentage), unsafe_allow_html=True
        # )


if __name__ == "__main__":
    main()
