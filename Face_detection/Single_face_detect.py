import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt



''' 
so in this file we are going to build face detection model using Voila Jones
'''

image = cv2.imread('single.jpeg',1)
print(f'the size of the original image:{image.shape}') #checking image size

image=cv2.resize(image,(520,520)) #image size to big so, we reducing         the image size


#reading viola jones math file from

math_file = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cordinates,num_of_faces = math_file.detectMultiScale2(image)
print(f'cordinates points:{cordinates}')

#using these cordinates points we are drawing rectangle box on image
x1 = cordinates[0][0]
y1 = cordinates[0][1]
x2 = cordinates[0][2]
y2 = cordinates[0][3]

pt1 = x1,y1
pt2 = x1+x2,y1+y2

font = cv2.FONT_HERSHEY_COMPLEX
text = 'Number of face detected ='  +str(len(cordinates))

#to make rectangle on image
cv2.rectangle(image,pt1,pt2,(0,0,255),2)
cv2.putText(image,text,(120,90),font,0.5,(0,0,0),1,1)

cv2.imshow('single_face',image)
cv2.waitKey()
cv2.destroyAllWindows()