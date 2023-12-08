# Description

[https://detect-my-tone.streamlit.app/](https://detect-my-tone.streamlit.app/)

This project combines a text sentiment analysis model with a facial emotion detection model to predict the overall tone of a text message.

The databases used to train these models and their licenses are as follow:

### Text Sentiment Analysis
["Emotions dataset for NLP"](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) by Praveen Govi, used under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

Unzip [emotions_dataset_for_nlp.zip](emotions_dataset_for_nlp.zip) if retraining model.

### Facial Emotion Detection
["Emotion Detection"](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data), used under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

Unzip [emotion_detection_dataset.zip](emotion_detection_dataset.zip) if retraining model.

## Instructions:

### To run UI locally:
  - [```python3 chat_ui.py```](chat_ui.py)

### To test models:
Text Sentiment Analysis:
  - [```python3 user_interface.py```](user_interface.py)

Facial Emotion Detection:
  - [```python3 user_interface_images.py```](user_interface_images.py)

### To train models:
Text Sentiment Analysis:
  - [```python3 text_sentiment_analysis.py```](text_sentiment_analysis.py)

Facial Emotion Detection:
  - [```python3 face_emotion_detection.py```](face_emotion_detection.py)



