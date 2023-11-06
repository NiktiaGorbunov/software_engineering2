import math
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image

DESIRED_HEIGHT = 360
DESIRED_WIDTH = 360

# Загрузка и отображение изображений объекта и фона
def display_images():
    st.write("1. Загрузите изображение объекта (например, человека):")
    obj_image = st.file_uploader("Загрузите объект", type=["jpg", "png"])

    st.write("2. Загрузите изображение фона:")
    bg_image = st.file_uploader("Загрузите фон", type=["jpg", "png"])

    return obj_image, bg_image

# Функция для изменения размера и отображения изображения
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

    st.image(img, use_column_width=True)

mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Основная часть Streamlit приложения
def main():
    st.title("Сегментация объекта и замена фона")
    st.write("Загрузите изображение объекта и фона, чтобы выполнить сегментацию и замену фона.")

    obj_image, bg_image = display_images()

    if obj_image is not None and bg_image is not None:
        with mp_selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
            try:
                obj_image = Image.open(obj_image)
                bg_image = Image.open(bg_image)

                st.write("Исходное изображение объекта:")
                st.image(obj_image)

                st.write("Исходное изображение фона:")
                st.image(bg_image)

                results = selfie_segmentation.process(
                    cv2.cvtColor(np.array(obj_image), cv2.COLOR_RGB2BGR)
                )

                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                output_image = np.where(condition, np.array(obj_image), np.array(bg_image))

                st.write("Результат сегментации и замены фона:")
                st.image(output_image, use_column_width=True)

            except Exception as e:
                st.error("Произошла ошибка: " + str(e))

if __name__ == "__main__":
    main()

