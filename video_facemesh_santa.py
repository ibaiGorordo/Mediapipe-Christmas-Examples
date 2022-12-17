# References:
# - Original model: https://google.github.io/mediapipe/solutions/face_mesh.html
# - Face swap example: https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python
# - Santa claus image: https://www.desertcart.com.cy/products/378368443-2021-christmas-santa-claus-full-head-latex-mask-realistic-face-human-mask-women-mrs-santa-claus-new-years-party-supplies


import cv2
from cap_from_youtube import cap_from_youtube
from utils.face_mesh_utils import SantaFace

# Image to swap face with (should have white background)
santa_image_url = "https://m.media-amazon.com/images/I/61KlHweQqxL.jpg"

# Initialize santaFace class
max_people = 1
santa_face = SantaFace(santa_image_url, max_people)

# Initialize video
cap = cap_from_youtube("https://youtu.be/rxxHvW0oNpU")
start_time = 60 + 50  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('output3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

cv2.namedWindow("santa face", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame
    ret, frame = cap.read()

    if not ret:
        continue

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    try:
        ret, santa_image = santa_face(frame)
    except:
        ret = False

    if ret:
        combined_image = cv2.hconcat([frame, santa_image])
    else:
        continue
        combined_image = cv2.hconcat([frame, frame])

    combined_image = cv2.resize(combined_image, (1280, 720))
    out.write(combined_image)
    cv2.imshow("santa face", combined_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()