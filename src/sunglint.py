import os
import cv2  
import numpy as np

def sunglint(filename): 
    img = cv2.imread(filename)  
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    ret, thresh_hold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY) 
    glint = np.sum(thresh_hold)/255
    return 100*(np.sum(thresh_hold)/255)/(thresh_hold.shape[0]*thresh_hold.shape[1])


 
  
cv2.imshow('Binary Threshold Image', thresh_hold) 

 

 
if cv2.waitKey(0) & 0xff == 25:  
    cv2.destroyAllWindows()