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

# 1 -> Load video from folder.
webcam = cv2.VideoCapture('v3.mp4')

# 2 -> Create Car Classifier by Using Pre-trained Car Classifier.
people_classifier = cv2.CascadeClassifier('people_haar_features.xml')

while True:

    # 3.1 -> Read the current Frame or image
    read_successful, frame = webcam.read()

    if read_successful:
            
        # 3 -> Generate Gray-Scale image.
        g_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4 -> Detect Car's Coordinates.
        people_coordinates = people_classifier.detectMultiScale(g_img1)

        # 5 -> Draw Rectangle on image
        for(x,y,w,h) in people_coordinates:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)


        cv2.imshow('Simple Pedestrain Tracking Application \n', frame)
        key = cv2.waitKey(1)

        if key == 81 or key ==113:
            break
         
    else:
        break

print("\n Padestrain's Coordinates : ",people_coordinates)
print("\n Code is Completed. \n")