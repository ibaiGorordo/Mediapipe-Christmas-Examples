# References: 
# - Original model: https://google.github.io/mediapipe/solutions/face_detection.html
# - reindeer image: https://pixabay.com/photos/reindeer-fruit-orange-fall-2805140/

import cv2
import numpy as np

from utils.reindeer_face_utils import ReindeerFace

# Initialize webcam
cap = cv2.VideoCapture(0)

reindeer_image_path = "images/reindeer.jpg"
reindeer_image = cv2.imread(reindeer_image_path)

# Landmark coordinates (left eye, right eye, nose) in the reindeer image
# If using a different image, select new coordinates running
# ReindeerFace.select_reindeer_landmark_pixels(reindeer_image)
reindeer_landmarks = np.array([[693, 1476], [1060, 1483], [865, 1783]], dtype=np.float32)
reindeer_face = ReindeerFace(reindeer_image, reindeer_landmarks)

cv2.namedWindow("reindeer face", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame
    ret, frame = cap.read()

    if not ret:
        continue

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    # Draw reindeer face
    output_img = reindeer_face.detect_and_draw_reindeer(frame)

    cv2.imshow("reindeer face", output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
