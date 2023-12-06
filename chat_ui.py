import streamlit as st
import pickle5 as pickle
import numpy as np
import re
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

# load vectorizer
vectorizer = pickle.load(open("text_sentiment_vectorizer.joblib", "rb"))
# load the model
loaded_model = pickle.load(open("text_sentiment_model.sav", "rb"))
# load labels
loaded_label_encoder = pickle.load(open("text_sentiment_label_encoder.joblib", "rb"))


def analyze_text_sentiment(user_input):
    # preprocess user input
    cleaned_input = re.sub(r"[^a-z0-9\s]", "", user_input.lower())

    input_vectorized = vectorizer.transform([cleaned_input])

    # predict sentiment
    prediction = loaded_model.predict(input_vectorized)

    # get label
    predicted_label = loaded_label_encoder.inverse_transform(prediction)

    return predicted_label[0]


def detect_facial_emotion(photo_path):
    # Load the saved model
    model = load_model("face_emotion_detection_model.keras")

    # Load and preprocess the new photo
    photo = load_img(photo_path, color_mode="grayscale", target_size=(48, 48))
    photo = img_to_array(photo)
    photo = np.expand_dims(photo, axis=0)  # Add batch dimension
    photo /= 255.0  # Normalize pixel values

    # Predict the emotion
    predictions = model.predict(photo)

    emotion_index = np.argmax(predictions)
    emotions = [
        "angry",
        "disgusted",
        "fearful",
        "happy",
        "neutral",
        "sad",
        "surprised",
    ]  # The order of emotions in the training set
    predicted_emotion = emotions[emotion_index]

    return predicted_emotion


def determine_overall_tone(facial_emotion, text_sentiment):
    # rules for combining sentiment and emotion
    rules = {
        ("angry", "anger"): "Anger",
        ("disgusted", "anger"): "Negative",
        ("fearful", "fear"): "Anxiety",
        ("happy", "joy"): "Elation",
        ("neutral", "love"): "Confidence",
        ("sad", "sadness"): "Melancholy",
        ("fearful", "sadness"): "Painful",
        ("surprised", "surprise"): "Amusement",
        ("sad", "surprise"): "Disappointment",
        ("angry", "sadness"): "Frustration",
        ("disgusted", "disgust"): "Repulsion",
        ("fearful", "surprise"): "Shock",
        ("happy", "love"): "Adoration",
        ("neutral", "joy"): "Contentment",
        ("sad", "anger"): "Resentment",
        ("surprised", "fear"): "Alarm",
        ("angry", "joy"): "Sarcasm",
        ("disgusted", "sadness"): "Contempt",
        ("fearful", "anger"): "Terror",
        ("happy", "surprise"): "Delight",
        ("neutral", "anger"): "Calmness",
        ("sad", "love"): "Sorrow",
        ("surprised", "joy"): "Excitement",
        # Could add more
    }

    # Determine the overall tone based on the rules
    if facial_emotion == "neutral":
        overall_tone = text_sentiment.capitalize()
    else:
        overall_tone = rules.get((facial_emotion, text_sentiment), "Complex")

    return overall_tone


# Title for the chat app
st.title("Text and Facial Emotion Detector")

text_bot = """
 [ ... ]
"""

face_bot = """
[^_^]
"""

tone_bot = """
 [ T ]
"""


imageCaptured = st.camera_input(f"{face_bot} Please take a photo to detect emotion")

if imageCaptured:
    user_message = st.text_input(
        f"{text_bot} Please enter a sentence to analyze sentiment: "
    )

    st.image(imageCaptured)

    if st.button("Send"):
        predicted_facial_emotion = detect_facial_emotion(imageCaptured)
        predicted_text_sentiment = analyze_text_sentiment(user_message)

        overall_tone = determine_overall_tone(
            predicted_facial_emotion, predicted_text_sentiment
        )

        st.write(f"{face_bot} Predicted emotion: **{predicted_facial_emotion}**")
        st.write(f"{text_bot} Predicted sentiment: **{predicted_text_sentiment}**")
        st.write(f"{tone_bot} Overall tone: **{overall_tone}**")
