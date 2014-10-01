import numpy as np
import cv2


cap = cv2.VideoCapture(1)
fgbg = cv2.BackgroundSubtractorMOG(30, 4, 0.1)

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(frame)

    cv2.imshow('mask', fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
