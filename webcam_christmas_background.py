# References: 
# - Original model: https://google.github.io/mediapipe/solutions/selfie_segmentation.html
# - Smooth image combination: https://stackoverflow.com/a/58445127

import cv2
from imread_from_url import imread_from_url
from utils.background_utils import BackgroundSwapper

# Initialize webcam
cap = cv2.VideoCapture(0)

# Read background image
background_image_url = "https://upload.wikimedia.org/wikipedia/commons/2/2a/ChristmasVillage2008.jpg"
background_image = imread_from_url(background_image_url)

# Initialize background swapper (0: default model, 1: landmark image optimized)
model_selection = 1
background_swapper = BackgroundSwapper(model_selection)

cv2.namedWindow("Christmas Background", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame
    ret, frame = cap.read()

    if not ret:
        continue

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    # Swap background
    combined_image = background_swapper.swap_background(frame, background_image)

    cv2.imshow('Christmas Background', combined_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
