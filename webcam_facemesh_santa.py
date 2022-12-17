# References:
# - Original model: https://google.github.io/mediapipe/solutions/face_mesh.html
# - Face swap example: https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python
# - Santa claus image: https://www.desertcart.com.cy/products/378368443-2021-christmas-santa-claus-full-head-latex-mask-realistic-face-human-mask-women-mrs-santa-claus-new-years-party-supplies


import cv2
from utils.face_mesh_utils import SantaFace

# Image to swap face with (should have white background)
santa_image_url = "https://m.media-amazon.com/images/I/61KlHweQqxL.jpg"

# Initialize santaFace class
max_people = 1
draw_santa = SantaFace(santa_image_url, max_people)

# Initialize webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow("santa face", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame
    ret, frame = cap.read()

    if not ret:
        continue

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    ret, santa_image = draw_santa(frame)

    if ret:
        cv2.imshow("santa face", santa_image)
    else:
        cv2.imshow("santa face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
