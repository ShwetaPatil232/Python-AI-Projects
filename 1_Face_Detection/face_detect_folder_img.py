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

# 3 -> Choose Image for face detection
img = cv2.imread('img2.jpg') #Choose Image from Folder.

# 4 -> Convert image into Gray-Scale
g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 5 -> Pass Gray-Scale image to Algorithem & Detect Faces Cordinates dots
face_coordinate = trained_face_data.detectMultiScale(g_img)

# 6 -> Draw Rectangle around face by using face coordinates.
for(x,y,w,h) in face_coordinate: # ( x,y,w,h) = face_coordinate[0] -> for Single Person Face Detection.
    cv2.rectangle(img, (x , y ), (x+w , y+h ), ( randrange(256), randrange(256), randrange(256)), 3)
    #  cv2.rectangle(img, pt1, pt2, color, thinkness)


cv2.imshow(' Simple Face Detection Application.', img)
cv2.waitKey() # Wait Till Key is Pressed

print('\n\n Image Face Coordinates : ', face_coordinate)
print("\n\n Code is Completed.")