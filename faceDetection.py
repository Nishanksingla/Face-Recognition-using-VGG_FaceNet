import cv2

import numpy as np
import base64
import requests

stream = cv2.VideoCapture(0)

if not stream.isOpened():
    print("Could not open camera..")
    sys.exit(1)

face_cascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")

while(True):
    ret, frame = stream.read()
    if ret == False:
        print "Error, unable to grab a frame from the camera"
        quit()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5,minSize=(80, 80))
    if len(faces)>0:
        x,y,w,h = faces[0]
        cv2.rectangle(frame,(x,y),(x+w-20,y+h+40),(255,0,0),2)
#        cv2.putText(frame,"Nishank",(x+w-100,y), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
        roi_color = frame[y:y+h, x:x+w]
#        cv2.imwrite("me.jpg",roi_color)
    else:
        print "No face detected"

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
stream.release()
cv2.destroyAllWindows()