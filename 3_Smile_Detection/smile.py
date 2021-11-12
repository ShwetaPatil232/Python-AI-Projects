import cv2
import numpy as np 

"""
------------------------------- Project Structure ------------------------------

Step 1 -> Find Faces in Our Images (Haar Algorithem. )

Step 2 -> Find smiles in those faces (Haar Algorithem. )

Step 3 -> Label the faces if it's smiling.
---------------------------------End Project ----------------------------------

"""

# 1  Load Faces & Smile classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

# 2 Grap Webcam
webcam = cv2.VideoCapture("v2.3gp")

while True:

    # 3 Read the current Frame/Image from Webcam
    success_frame , frame = webcam.read()

    if not success_frame:
        break

    else:

        # 4 Convert to Gray-Scale
        g_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 5 Get Face-coordinates
        face_coordinates = face_classifier.detectMultiScale(g_img,)

        for (x,y,w,h) in face_coordinates:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

            # find Smile on face area.
            the_face = frame[y:y+h , x:x+w ] # Crop Face sized image to find smile within it
            g_face = cv2.cvtColor(the_face, cv2.COLOR_RGB2GRAY) # convert face to Gray-Scale
            smile_coordinates = smile_classifier.detectMultiScale(g_face, scaleFactor = 1.7, minNeighbors = 20)
            # draw Reactangles around Smile
            
            for (x_2,y_2,w_2,h_2) in smile_coordinates:
                cv2.rectangle(the_face, (x_2,y_2), (x_2 + w_2, y_2 + h_2), (0,0,255), 3) 

            # Label this face as smiling 
            if len(smile_coordinates) > 0:
                cv2.putText(frame, "Smiling", (x, y+h+40), fontFace= cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255,0,0))

        cv2.imshow("Simple Smile Detection App", frame)
        key = cv2.waitKey(1)
        if key ==81 or key ==113:
            break


webcam.release()
cv2.destroyAllWindows()
print("\n Code is Complete. \n")