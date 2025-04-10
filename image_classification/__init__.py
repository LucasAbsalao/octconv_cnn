import cv2
import numpy as np

PATH_IMAGES = "../image/"

img = cv2.imread(PATH_IMAGES+"tigre.jpg")

cv2.imshow("teste", img)
cv2.waitKey(0)
cv2.destroyWindow("teste")