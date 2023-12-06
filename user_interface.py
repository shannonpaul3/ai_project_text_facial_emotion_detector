import pickle5 as pickle

# Collect user input
user_input = input("Please enter a sentence to analyze sentiment: ")

# load vectorizer
vectorizer = pickle.load(open("text_sentiment_vectorizer.joblib", "rb"))
input_vectorized = vectorizer.transform([user_input])

# load the model
loaded_model = pickle.load(open("text_sentiment_model.sav", "rb"))

# predict sentiment
prediction = loaded_model.predict(input_vectorized)

# load labels
loaded_label_encoder = pickle.load(open("text_sentiment_label_encoder.joblib", "rb"))

# get label
predicted_label = loaded_label_encoder.inverse_transform(prediction)

# Display the result
print(f"Possible sentiments: {loaded_label_encoder.classes_}")
print(f"The predicted sentiment is: {predicted_label[0]}")
