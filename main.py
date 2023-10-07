import cv2 as cv
import numpy as np

#HOG
hog=cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
cv.startWindowThread()
# open webcam video stream
cap=cv.VideoCapture(1)
# the output will be written to output.avi
out=cv.VideoWriter('output.avi',cv.VideoWriter_fourcc(*'MJPG'),15,(640,480))
out = cv.VideoWriter(
    'output.avi',
    cv.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
while(True):
    ret, frame =cap.read()
    frame=cv.resize(frame,(640,480))
    gray=cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weigths=hog.detectMultiScale(frame,winStride=(8,8))
    boxes=np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])
    for (xA, yA, xB, yB) in boxes:
        cv.rectangle(frame,(xA,yA),(xB,yB),(0,255,0),2)
    out.write(frame.astype('uint8'))
    cv.imshow('frame',frame)
    if cv.waitKey(1)& 0xFF == ord('q'):
        break
cap.release()
out.release()
cv.destroyAllWindows()
cv.waitKey(1)
