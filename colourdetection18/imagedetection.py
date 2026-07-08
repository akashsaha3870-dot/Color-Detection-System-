import numpy as np
import cv2

# Read the image
imageFrame = cv2.imread("imagergb.jpg")
if imageFrame is None:
    print("Error: Image not found. Check file path!")
    exit()

# Convert BGR to HSV
hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

# Red
red_lower = np.array([136, 87, 111], np.uint8)
red_upper = np.array([180, 255, 255], np.uint8)
red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

# Green
green_lower = np.array([25, 52, 72], np.uint8)
green_upper = np.array([102, 255, 255], np.uint8)
green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

# Blue
blue_lower = np.array([94, 80, 2], np.uint8)
blue_upper = np.array([120, 255, 255], np.uint8)
blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

# Kernel
kernel = np.ones((5, 5), "uint8")
red_mask = cv2.dilate(red_mask, kernel)
green_mask = cv2.dilate(green_mask, kernel)
blue_mask = cv2.dilate(blue_mask, kernel)

# Red contours
contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 300:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(imageFrame, "Red Colour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Green contours
contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 300:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(imageFrame, "Green Colour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Blue contours
contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 300:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(imageFrame, "Blue Colour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Show image
cv2.imshow("Color Detection Result", imageFrame)
cv2.waitKey(0)
cv2.destroyAllWindows()
input("Press Enter to exit...")





