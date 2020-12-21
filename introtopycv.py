import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
EyeDetect=cv2.CascadeClassifier('haarcascade_eye.xml');
cam=cv2.VideoCapture(0);

while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift=cv2.SIFT()
    kp= sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp)
    cv2.imwrite('sift_keypoints.jpg',img)
    faces=faceDetect.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=4, minSize=(100,100));
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),3)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=EyeDetect.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0),5);
       
        cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
         break;
cam.release()
cv2.destroyAllWindows()
        








                       
