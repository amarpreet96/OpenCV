#images are stored in numpy array so the library is imported
import numpy as np
#cv2 is imported from openCV to use its attributes
import cv2



#Face haarcascade file is used to create a face cascade for face detection
face_cascade = cv2.CascadeClassifier('face.xml')

#eye haarcascade file  is used to create eye cascade for eye detection
eye_cascade = cv2.CascadeClassifier('eye.xml')

#nose haarcascade file is used to create nose cascade for nose detection
nose_cascade = cv2.CascadeClassifier('Nariz.xml')

#mouth haarcascade file is used  to create mouth cascade for mouth detection
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')

#checking for haarcascade file if present or not
if nose_cascade.empty():
  raise IOError('Unable to load the nose cascade classifier xml file')

#function used to capture video from webcam
cap = cv2.VideoCapture(0)

#while loop start for video input for True
while 1:
    #creating Frame for given video input through webcam
    ret, img = cap.read()
   
    #converting to grayscale to ease the process of detection
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #For each face detectMultiScale function returns a rectangle surrounding the face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #rectangle REGION OF INTEREST(ROI) is specified and values are passed in rectangle function 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#Blue coloured rectangle
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        #eye detector works for given ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)#green cloured rectangle
            
    #nose detector works for given ROI      
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in nose_rects:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3)#red coloured rectangle
        break

    
    #mouse detector works for given ROi
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x,y,w,h) in mouth_rects:
        y = int(y - 0.15*h)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 3)
        break
    cv2.imshow('img',img)#displaying the frame created with detection performed
    k = cv2.waitKey(30) & 0xff#wait key is set until program has finish its work 
    if k == 27:
        break

cap.release()#releasing video input after it work is completed
cv2.destroyAllWindows()#destroying all windows after execution of program