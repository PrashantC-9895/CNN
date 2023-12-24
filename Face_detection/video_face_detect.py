import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt



''' 
so in this file we are going to build face detection model using Voila Jones
'''
math_file = cv2.CascadeClassifier('C://Users//pcrid//deep learninng//14.10.2023//haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('C://Users/pcrid//deep learninng//14.10.2023//obama.mp4')
while cap.read():
    res , frame = cap.read()
    if res == True :
        print(f'The shape of the frame : {frame.shape}')  # checking image size
        frame = cv2.resize(frame, (520, 520))  # image size to big so, we reducing the image size
        cordinates, num_of_faces = math_file.detectMultiScale2(frame)
        print(f'cordinates :{cordinates}')
        if len(cordinates) > 0 :
            # using these cordinates points we are drawing rectangle box on image
            x1 = cordinates[0][0]
            y1 = cordinates[0][1]
            x2 = cordinates[0][2] + x1
            y2 = cordinates[0][3] + y1

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            text = "face : Obama"
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(frame,text,(x1,y1-5),font,0.5,(0,0,255),1,cv2.LINE_AA)
            cv2.imshow('frame',frame)
        else:
            pass
        if cv2.waitKey(1)% 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


