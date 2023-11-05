import math
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image


DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


def resize_and_show(image):
    height, weight = image.shape[:2]
    if height < weight:
        img = cv2.resize(
            image, (DESIRED_WIDTH, math.floor(height / (weight / DESIRED_WIDTH)))
        )
    else:
        img = cv2.resize(
            image, (math.floor(weight / (height / DESIRED_HEIGHT)), DESIRED_HEIGHT)
        )

    cv2.imshow("img", img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Blur the image background based on the segementation mask.
with mp_selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
    # Convert the BGR image to RGB and process it with MediaPipe Selfie Segmentation.
    try:
        files = st.file_uploader(
            "Upload images", type=["jpg", "png"], accept_multiple_files=True
        )

        for file in files:
            image = Image.open(file)
            st.image(image)

        files_path = list(map(lambda file: file.name, files))
        # Сначала загружаем объект, затем фон
        obj_image = cv2.imread("datasets/" + files_path[0])
        bg_image = cv2.imread("backgrounds/" + files_path[1])

        results = selfie_segmentation.process(
            cv2.cvtColor(obj_image, cv2.COLOR_BGR2RGB)
        )

        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        output_image = np.where(condition, obj_image, bg_image)

        # resize_and_show(output_image)
        st.image(output_image)

    except IndexError:
        pass
