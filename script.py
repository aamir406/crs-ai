import cv2
import numpy as np

img = cv2.imread('images/test_image.png')

test = img

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
mask_red = cv2.bitwise_or(mask1, mask2)
# mask_red = cv2.bitwise_and(img, img, mask=mask)

contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in contours:
    approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
    x, y, w, h = cv2.boundingRect(approx)

    if len(approx) == 4 and w*h > 5000:
        boxes.append((x, y, w, h))
        cv2.rectangle(test, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Detected Boxes", test)
cv2.waitKey(3000)
cv2.destroyAllWindows()





# cv2.imwrite("target.png", target)


print(img)