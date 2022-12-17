# Ref: https://google.github.io/mediapipe/solutions/selfie_segmentation.html
import mediapipe as mp
import cv2
import numpy as np


class BackgroundSwapper:
    def __init__(self, model_selection=0):
        self.model_selection = model_selection
        self.selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection)

    def swap_background(self, frame, background):
        background_mask = self.get_background(frame)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2RGB)
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
        return np.uint8(frame * background_mask + background * (1 - background_mask))

    def get_background(self, frame):
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img.flags.writeable = False
        output = self.selfie_segmentation.process(input_img)
        return output.segmentation_mask

    def get_foreground(self, frame):
        return 1 - self.get_background(frame)


if __name__ == '__main__':
    from imread_from_url import imread_from_url

    # Read input image
    input_image_url = "https://upload.wikimedia.org/wikipedia/commons/1/1b/Jos%C3%A9_El%C3%ADas_Moreno_in_Santa_Claus_%281959%29_%284%29.png"
    input_image = imread_from_url(input_image_url)

    # Read background image
    background_image_url = "https://upload.wikimedia.org/wikipedia/commons/2/2a/ChristmasVillage2008.jpg"
    background_image = imread_from_url(background_image_url)

    # Initialize background swapper (0: default model, 1: landmark image optimized)
    model_selection = 1
    background_swapper = BackgroundSwapper(model_selection)

    # Swap background
    combined_image = background_swapper.swap_background(input_image, background_image)

    # Display combined image
    cv2.namedWindow("Combined Image", cv2.WINDOW_NORMAL)
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)
