import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers.legacy import Adam

# Set up the ImageDataGenerators for training and testing
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load images from the directories
train_generator = train_datagen.flow_from_directory(
    "emotion_detection_dataset/train",
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
)

test_generator = test_datagen.flow_from_directory(
    "emotion_detection_dataset/test",
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
)

num_filters = 8
filter_size = 3
pool_size = 2
num_classes = 7  # You have 7 emotions

# Build the model.
model = Sequential(
    [
        Conv2D(num_filters, filter_size, input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(128, activation="relu"),  # Added a dense layer for better performance
        Dense(num_classes, activation="softmax"),
    ]
)

# Compile the model.
model.compile(
    optimizer=Adam(learning_rate=0.001),  # You can adjust the learning rate as needed
    loss="categorical_crossentropy",
    metrics=["accuracy", "Precision", "Recall"],
)

# Train the model using the generators
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
)

# Save the model
model.save("face_emotion_detection_model.keras")

# Evaluate the model on the test set
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
    test_generator, steps=test_generator.samples // test_generator.batch_size
)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
