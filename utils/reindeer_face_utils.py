# References:
# - Mouse click coordinates: https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture/28330835

import cv2
import numpy as np
import mediapipe as mp


class ReindeerFace:

    def __init__(self,
                 reindeer_image,
                 reindeer_landmarks,
                 model_selection=0,
                 min_detection_confidence=0.7):

        # Initialize face detection (0: small model for distance < 2m,
        #                           1: full range model for distance < 5m)
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=model_selection,
                                                                        min_detection_confidence=min_detection_confidence)

        self.reindeer_image = reindeer_image
        self.reindeer_landmarks = reindeer_landmarks

    def __call__(self, image):

        return self.detect_and_draw_reindeer(image)

    def detect_and_draw_reindeer(self, image):
        # Detect face
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image.flags.writeable = False
        detections = self.face_detection.process(input_image).detections



        # Draw reindeer
        return self.draw_reindeer(image, self.reindeer_image, detections)

    def draw_reindeer(self, image, reindeer_img, detections):
        input_height, input_width = image.shape[:2]

        output_img = image.copy()

        if detections:
            for detection in detections:
                face_coordinates = np.array([[detection.location_data.relative_keypoints[i].x * input_width,
                                              detection.location_data.relative_keypoints[i].y * input_height]
                                             for i in [0, 1, 2]], dtype=np.float32)
                M = cv2.getAffineTransform(self.reindeer_landmarks, face_coordinates)
                transformed_reindeer = cv2.warpAffine(reindeer_img, M, (input_width, input_height))

                transformed_reindeer_mask = cv2.inRange(transformed_reindeer, (0, 0, 50), (250, 215, 215))

                # Erode mask to remove artifacts
                erode_kernel = np.ones((5, 5), np.uint8)
                transformed_reindeer_mask = cv2.erode(transformed_reindeer_mask, erode_kernel, iterations=1) > 0

                output_img[transformed_reindeer_mask] = transformed_reindeer[transformed_reindeer_mask]

        return output_img

    @staticmethod
    def draw_mouseclick_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(reindeer_image, (x, y), 30, (0, 0, 255), -1)
            print(f"[{x}, {y}]")

    @staticmethod
    def select_reindeer_landmark_pixels(image):
        # Order: Left eye, right eye, nose
        cv2.namedWindow("Double click on left eye, right eye and nose", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Double click on left eye, right eye and nose", ReindeerFace.draw_mouseclick_circle)

        while True:
            cv2.imshow("Double click on left eye, right eye and nose", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    # Read reindeer image
    reindeer_image_path = "../images/reindeer.jpg"
    reindeer_image = cv2.imread(reindeer_image_path)

    ReindeerFace.select_reindeer_landmark_pixels(reindeer_image)
