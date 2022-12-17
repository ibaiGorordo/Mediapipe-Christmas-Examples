# Ref: https://github.com/rcsmit/python_scripts_rcsmit/blob/master/extras/Gal_Gadot_by_Gage_Skidmore_4_5000x5921_annotated_white_letters.jpg
import cv2
import mediapipe as mp
import numpy as np
from imread_from_url import imread_from_url
from utils.background_utils import BackgroundSwapper


class SantaFace:

    def __init__(self, santa_image_url, max_people=1, detection_confidence=0.3):

        self.santa_face_coordinates = None
        self.santa_image = None
        self.santa_mask = None
        self.face_mesh = None

        self.vertices = np.loadtxt("utils/vertices.txt", dtype=np.int32)
        self.initialize_model(max_people, detection_confidence)

        self.prepare_santa_image(santa_image_url)

    def __call__(self, image):

        return self.detect_and_draw_santa(image)

    def detect_and_draw_santa(self, image):

        landmarks = self.detect_face_mesh(image)

        if not landmarks:
            return False, None

        return True, self.draw_santa_face(image, landmarks)

    def initialize_model(self, max_people, detection_confidence):

        # Initialize face mesh detection (0: default model, 1: landmark image optimized)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=max_people,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=detection_confidence,
                                                         min_tracking_confidence=detection_confidence)

    def prepare_santa_image(self, santa_image_url):
        # Read santa image
        self.santa_image = imread_from_url(santa_image_url)

        # Detect santa mesh for later use
        self.santa_face_coordinates = self.detect_face_mesh(self.santa_image)[0]

        # Remove background from santa image
        background_swapper = BackgroundSwapper(model_selection=1)
        background_mask = background_swapper.get_background(self.santa_image)
        self.santa_image[background_mask < 0.7] = [0, 0, 0]

    def detect_face_mesh(self, image):

        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image.flags.writeable = False
        results = self.face_mesh.process(input_image)

        if results.multi_face_landmarks is None:
            return None

        img_height, img_width, _ = image.shape
        return self.scale_landmarks(results.multi_face_landmarks, img_width, img_height)

    def draw_santa_face(self, img, landmarks):

        converted_image = np.zeros(img.shape, dtype=np.uint8)
        for face_landmarks in landmarks:

            face_landmarks[face_landmarks < 0] = 0
            for triangle_id in range(0, len(self.vertices), 3):
                # Get triangle vertex coordinates
                corner1_id = self.vertices[triangle_id][0]
                corner2_id = self.vertices[triangle_id + 1][0]
                corner3_id = self.vertices[triangle_id + 2][0]

                santa_pix_coords = self.santa_face_coordinates[[corner1_id, corner2_id, corner3_id], :]
                face_pix_coords = face_landmarks[[corner1_id, corner2_id, corner3_id], :]

                # Crop the images to the section with the triangles
                ex_x, ex_y, ex_w, ex_h = cv2.boundingRect(santa_pix_coords)
                face_x, face_y, face_w, face_h = cv2.boundingRect(face_pix_coords)
                cropped_santa = self.santa_image[ex_y:ex_y + ex_h, ex_x:ex_x + ex_w]

                # Update the triangle coordinates for the cropped image
                santa_pix_crop_coords = santa_pix_coords.copy()
                face_pix_crop_coords = face_pix_coords.copy()
                santa_pix_crop_coords[:, 0] -= ex_x
                santa_pix_crop_coords[:, 1] -= ex_y
                face_pix_crop_coords[:, 0] -= face_x
                face_pix_crop_coords[:, 1] -= face_y

                # Get the mask for the triangle in the cropped face image
                cropped_face_mask = np.zeros((face_h, face_w), np.uint8)
                triangle = (np.round(np.array([face_pix_crop_coords]))).astype(int)

                cv2.fillConvexPoly(cropped_face_mask, triangle, 255)

                # Warp cropped santa triangle into the cropped face triangle
                warp_mat = cv2.getAffineTransform(santa_pix_crop_coords.astype(np.float32),
                                                  face_pix_crop_coords.astype(np.float32))
                warped_triangle = cv2.warpAffine(cropped_santa, warp_mat, (face_w, face_h))
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_face_mask)

                # Put the warped triangle into the destination image
                cropped_new_face = converted_image[face_y:face_y + face_h, face_x:face_x + face_w]
                cropped_new_face_gray = cv2.cvtColor(cropped_new_face, cv2.COLOR_BGR2GRAY)
                _, non_filled_mask = cv2.threshold(cropped_new_face_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=non_filled_mask)
                cropped_new_face = cv2.add(cropped_new_face, warped_triangle)
                converted_image[face_y:face_y + face_h, face_x:face_x + face_w] = cropped_new_face

            converted_image = self.add_santa_image(converted_image, face_landmarks)

        # Add the santa mask to the webcam feed
        converted_image_gray = cv2.cvtColor(converted_image, cv2.COLOR_BGR2GRAY)
        _, non_drawn_mask = cv2.threshold(converted_image_gray, 1, 255, cv2.THRESH_BINARY_INV)
        img = cv2.bitwise_and(img, img, mask=non_drawn_mask)
        converted_image = cv2.add(converted_image, img)

        return converted_image

    def add_santa_image(self, converted_image, face_landmarks):

        img_height, img_width, _ = converted_image.shape
        face_coord_ind = [234, 447, 152]
        face_triangle_coord = face_landmarks[face_coord_ind, :].astype(np.float32)
        santa_triangle_coord = self.santa_face_coordinates[face_coord_ind, :].astype(np.float32)

        M = cv2.getAffineTransform(santa_triangle_coord, face_triangle_coord)
        transformed_santa = cv2.warpAffine(self.santa_image, M, (img_width, img_height))

        transformed_santa_mask = cv2.inRange(transformed_santa, (1, 1, 1), (254, 254, 254))

        # Erode mask to remove artifacts
        erode_kernel = np.ones((3, 3), np.uint8)
        transformed_santa_mask = cv2.erode(transformed_santa_mask, erode_kernel, iterations=1)
        converted_image_gray = cv2.cvtColor(converted_image, cv2.COLOR_BGR2GRAY)
        _, non_drawn_mask = cv2.threshold(converted_image_gray, 1, 255, cv2.THRESH_BINARY_INV)

        # FIll mask for mouth and eye holes
        left_eye_ids = [113, 224, 222, 190, 22, 110, 113]
        right_eye_ids = [464, 442, 444, 446, 254, 252, 464]
        mouth_ids = [57, 37, 267, 287, 314, 84, 57]
        non_drawn_mask = cv2.fillConvexPoly(non_drawn_mask, face_landmarks[left_eye_ids, :], 0)
        non_drawn_mask = cv2.fillConvexPoly(non_drawn_mask, face_landmarks[right_eye_ids, :], 0)
        non_drawn_mask = cv2.fillConvexPoly(non_drawn_mask, face_landmarks[mouth_ids, :], 0)

        transformed_santa_mask = cv2.bitwise_and(transformed_santa_mask,
                                                 non_drawn_mask) > 0

        converted_image[transformed_santa_mask] = transformed_santa[transformed_santa_mask]
        return converted_image

    @staticmethod
    def scale_landmarks(landmarks, img_width, img_height):

        converted_landmarks = []
        for face_landmarks in landmarks:
            converted_face_landmarks = []
            for landmark in face_landmarks.landmark:
                converted_face_landmarks.append([landmark.x * img_width, landmark.y * img_height])
            converted_landmarks.append(np.array(converted_face_landmarks, dtype=np.int32))
        return converted_landmarks