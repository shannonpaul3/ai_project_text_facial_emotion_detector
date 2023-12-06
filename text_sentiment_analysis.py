import numpy as np
import pickle5 as pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# read in txt files
training_data = np.loadtxt(
    "emotions_dataset_for_nlp/train.txt", dtype=str, delimiter=";"
)
validation_data = np.loadtxt(
    "emotions_dataset_for_nlp/val.txt", dtype=str, delimiter=";"
)
testing_data = np.loadtxt("emotions_dataset_for_nlp/test.txt", dtype=str, delimiter=";")


# split data into samples and labels
training_samples = training_data[:, 0]
training_labels = training_data[:, 1]

validation_samples = validation_data[:, 0]
validation_labels = validation_data[:, 1]

testing_samples = testing_data[:, 0]
testing_labels = testing_data[:, 1]


# generate 1-gram (unigram) models
vectorizer = CountVectorizer(ngram_range=(1, 1))
X_train = vectorizer.fit_transform(training_samples)
X_val = vectorizer.transform(validation_samples)
X_test = vectorizer.transform(testing_samples)


# encode labels
le = LabelEncoder()
y_train = le.fit_transform(training_labels)
y_val = le.transform(validation_labels)
y_test = le.transform(testing_labels)

# select model
model = MultinomialNB()

# train model
model.fit(X_train, y_train)

# evaluate model
y_val_pred = model.predict(X_val)
# print(classification_report(y_val, y_val_pred, zero_division=0))

# test model
y_test_pred = model.predict(X_test)
# print(classification_report(y_test, y_test_pred, zero_division=0))

# Save the model to a file
pickle.dump(model, open("text_sentiment_model.sav", "wb"))

# Export the LabelEncoder to a file
pickle.dump(le, open("text_sentiment_label_encoder.joblib", "wb"))

# Export the vectorizer to a file
pickle.dump(vectorizer, open("text_sentiment_vectorizer.joblib", "wb"))
