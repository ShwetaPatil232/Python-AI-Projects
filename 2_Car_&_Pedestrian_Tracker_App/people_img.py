import cv2

""" -------------------- Part 1 : Car  Tracking System Structure (Haar Features.) ----------------------------
Step 1 -> Get a lot of car images & Load them.
Step 2 -> Make them all black & while (Grayscale Image.)
Step 3 -> Train the algorithem to detect cars.

-------------------- Part 2 : Pedestrian  Tracking System Structure  ----------------------------------------
Step 1 -> Get a lot of pedestrian images & Load them.
Step 2 -> Make them all black & while (Grayscale Image.)
Step 3 -> Train the algorithem to detect pedestrian.

-------------------------------------------- End Project. --------------------------------------------------
"""

# 1 -> Load Image from folder.
img1 = cv2.imread('people11.png')

# 2 -> Create Car Classifier by Using Pre-trained Car Classifier.
people_classifier = cv2.CascadeClassifier('people_haar_features.xml')

# 3 -> Generate Gray-Scale image.
g_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# 4 -> Detect Car's Coordinates.
people_coordinates = people_classifier.detectMultiScale(g_img)

# 5 -> Draw Rectangle on image
for(x,y,w,h) in people_coordinates:
    cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0),3)


cv2.imshow('Simple Pedestrain Tracking Application \n', img1)
cv2.waitKey()

print("\n Padestrain's Coordinates : ",people_coordinates)
print("\n Code is Completed. \n")