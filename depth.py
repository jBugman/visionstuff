import numpy as np
import cv2


cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, 16, 7)

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    imgL = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, frame2 = cap2.read()
    frame2 = cv2.flip(frame2, 1)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    imgR = cv2.resize(imgR, None, fx=0.666666, fy=0.666666, interpolation=cv2.INTER_CUBIC)


    disp = stereo.compute(imgL, imgR, disptype=cv2.CV_32F)
    norm_coeff = 255 / disp.max()
    cv2.imshow('depth', disp * norm_coeff / 255)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()