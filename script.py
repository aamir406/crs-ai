import cv2
import numpy as np

img = cv2.imread('images/test_page.png')

cv2.imshow("pic", img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert to HSV

# Set hue ranges for red colour
lower_red1 = np.array([0, 100, 100])     # dark red
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])   # bright red
upper_red2 = np.array([180, 255, 255])

# define masks for both ranges
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# apply final masks
mask = cv2.bitwise_or(mask1, mask2)
target = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite("target.png", target)


print(img)