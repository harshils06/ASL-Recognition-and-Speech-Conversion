import cv2
import numpy as np
import pyttsx3

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare data generator for standardizing frames before sending them into the model.
data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

# Loading the model.
MODEL_NAME = 'models/asl_alphabet.h5'
model = load_model(MODEL_NAME)

# Setting up the input image size and frame crop size.
IMAGE_SIZE = 200
CROP_SIZE = 400

# Creating list of available classes stored in classes.txt.
classes_file = open("classes.txt")
classes_string = classes_file.readline()
classes = classes_string.split()
classes.sort()  # The predict function sends out output in sorted order.

# Preparing cv2 for webcam feed
cap = cv2.VideoCapture(0)

# Setting up text-to-speech engine
engine = pyttsx3.init()

predict = True

while(True):
    # Capture frame-by-frame.
    ret, frame = cap.read()

    # Target area where the hand gestures should be.
    cv2.rectangle(frame, (0, 0), (CROP_SIZE, CROP_SIZE), (0, 255, 0), 3)
    
    # Preprocessing the frame before input to the model.
    cropped_image = frame[0:CROP_SIZE, 0:CROP_SIZE]
    resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
    reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

    if predict:
        # Predicting the frame.
        prediction = np.array(model.predict(frame_for_model))
        predicted_class = classes[prediction.argmax()]      # Selecting the max confidence index.

        # Preparing output based on the model's confidence.
        prediction_probability = prediction[0, prediction.argmax()]
        if predicted_class != "nothing":
            text_output = ""
            if prediction_probability > 0.9:
                # High confidence.
                text_output = predicted_class
                predict=False
            
        # Speak the predicted class
        if text_output != "":
            engine.say(text_output)
            engine.runAndWait()
    cv2.putText(frame, text_output, (10, 450), 1, 2, (0, 255, 255), 2, cv2.LINE_AA)
    # Display the image with prediction.
    cv2.imshow('frame', frame)

    # Press q to quit
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('r'):
        predict = True

# When everything done, release the capture.
cap.release()
cv2.destroyAllWindows()
