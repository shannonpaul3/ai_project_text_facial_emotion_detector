from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2 as cv2

# Initialize the camera
cam = cv2.VideoCapture(0)  # 0 is usually the built-in webcam on MacBook

# Set the desired square resolution, e.g., 480x480
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Press Space to capture")

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Crop the frame to a square (if the captured frame is not already square)
    height, width = frame.shape[:2]
    min_dim = min(height, width)
    top_left_x = (width - min_dim) // 2
    top_left_y = (height - min_dim) // 2
    square_frame = frame[
        top_left_y : top_left_y + min_dim, top_left_x : top_left_x + min_dim
    ]

    # Display the resulting square frame
    cv2.imshow("Press Space to capture", square_frame)

    # Wait for keypress
    k = cv2.waitKey(1)

    if k % 256 == 27:  # ESC key to exit
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:  # SPACE key to take a photo
        # Save the captured square frame as an image file
        img_name = "input_image.jpg"
        cv2.imwrite(img_name, square_frame)
        print(f"{img_name} captured!")
        break

# When everything is done, release the capture and close windows
cam.release()
cv2.destroyAllWindows()

# Load the saved model
model = load_model("face_emotion_detection_model.keras")

# Load and preprocess the new photo
photo = load_img(
    "input_image",
    color_mode="grayscale",
    target_size=(48, 48),
)
photo = img_to_array(photo)
photo = np.expand_dims(photo, axis=0)  # Add batch dimension
photo /= 255.0  # Normalize pixel values

# Predict the emotion
predictions = model.predict(photo)

# Predictions will be a list of probabilities
emotion_probability = np.max(predictions)
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

print(
    f"Predicted emotion: {predicted_emotion} with a probability of {emotion_probability:.2f}"
)
