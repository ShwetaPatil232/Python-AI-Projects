import cv2
from random import randrange

""" --------------------Face Detection Project Structure  ----------------------------
Step 1 -> Get Tones of Faces & Load them.
Step 2 -> Make them all black & while (Grayscale Image.)
Step 3 -> Train the algorithem to detect faces.
------------------------------- End Project. --------------------------------
"""
# 1 -> Download File -> https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

# 2 -> Load Pretrained Data on face frontals from opencv (haar cascade algorithrm.)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 3 -> Capture Image from Webcam
webcam = cv2.VideoCapture(0)

while True:

    # Read the current Frame or image
    success_bool , frame = webcam.read()

    # 4 -> Convert image into Gray-Scale
    g_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5 -> Pass Gray-Scale image to Algorithem & Detect Faces Cordinates dots
    face_coordinate = trained_face_data.detectMultiScale(g_img)

    # 6 -> Draw Rectangle around face by using face coordinates.
    for(x,y,w,h) in face_coordinate:
        cv2.rectangle(frame, (x , y ), (x+w , y+h ), ( randrange(256), randrange(256), randrange(256)), 3)
    
    cv2.imshow('Clever Programmer Face Detection Tutorial', frame)
    key = cv2.waitKey(1) # Wait Till Key is Pressed

    # 7 -> Stop if Q is Pressed
    if key == 81 or key == 113:
        break

# Relaese the VideoCapture Object
webcam.release()
print("\n Code is Completed. \n")
