import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt



''' 
so in this file we are going to build face detection model using Voila Jones
'''

image = cv2.imread('C://Users//pcrid//deep learninng//14.10.2023//multiple.jpg',1)
print(f'The shape of the image : {image.shape}') #checking image size

image=cv2.resize(image,(520,520)) #image size to big so, we reducing the image size


#reading viola jones math file from

math_file = cv2.CascadeClassifier('C://Users//pcrid//deep learninng//14.10.2023//haarcascade_frontalface_default.xml')
cordinates,num_of_faces = math_file.detectMultiScale2(image)
print(f'cordinates :{cordinates}')
print(f'length of cordiantes:{len(cordinates)}')

cordinates_1 = cordinates
for i in range (len(cordinates)):
    cordinates_1 = cordinates[i]
    #using these cordinates points we are drawing rectangle box on image
    x1 = cordinates_1[0]
    y1 = cordinates_1[1]
    x2 = cordinates_1[2] + x1
    y2 = cordinates_1[3] + y1

    # to make rectangle on image
    cv2.rectangle(image, (x1,y1),(x2,y2), (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_COMPLEX
    text = f'face :{i}'
    cv2.putText(image, text, (x1,y1-5), font, 0.5, (0, 255, 0), 1, 1)




cv2.imshow('multiple.jpg',image)
cv2.waitKey()
cv2.destroyAllWindows()