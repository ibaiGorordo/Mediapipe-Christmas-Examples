# References: 
# - Original model: https://google.github.io/mediapipe/solutions/selfie_segmentation.html
# - Smooth image combination: https://stackoverflow.com/a/58445127

import cv2
from imread_from_url import imread_from_url
from cap_from_youtube import cap_from_youtube
from utils.background_utils import BackgroundSwapper

# Initialize video
cap = cap_from_youtube("https://youtu.be/Bw5yS0f1bjc", resolution="720p")
start_time = 5  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

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

