import cv2
import math
import numpy as np
import mediapipe as mp

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow('img',img)

  cv2.waitKey(0)

  # It is for removing/deleting created GUI window from screen
  # and memory
  cv2.destroyAllWindows()

images = ['datasets/cloun.jpg']
# Read images with OpenCV.
images = {name: cv2.imread(name) for name in images}
# Preview the images.
for name, image in images.items():
  print(name)
  resize_and_show(image)

mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Blur the image background based on the segementation mask.
with mp_selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
    for name, image in images.items():
        # Convert the BGR image to RGB and process it with MediaPipe Selfie Segmentation.
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        bg_image = cv2.imread('backgrounds/back_test.jpg')
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        output_image = np.where(condition, image, bg_image)

        print(f'Blurred background of {name}:')
        resize_and_show(output_image)