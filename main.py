import numpy as np
import cv2


BASE_PATH = '/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/'

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(BASE_PATH + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(BASE_PATH + 'haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier(BASE_PATH + 'haarcascade_smile.xml')


while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    height, width, depth = frame.shape
    fov_w, fov_h = int(width * 0.33), int(height * 0.66)
    fov_x, fov_y = (width - fov_w) / 2, (height - fov_h) / 2,

    cv2.rectangle(frame, (fov_x, fov_y), (fov_x + fov_w, fov_y + fov_h), (0, 0, 255), 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    search_field = gray[fov_y: fov_y + fov_h, fov_x: fov_x + fov_w]
    search_field_color = frame[fov_y: fov_y + fov_h, fov_x: fov_x + fov_w]
    faces = face_cascade.detectMultiScale(search_field, 1.2, 7)

    for (x, y, w, h) in faces:
        cv2.rectangle(search_field_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = search_field[y: y + h, x: x + w]
        roi_color = search_field_color[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 7)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        # smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=20, minSize=(20, 20), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        # smile = smile_cascade.detectMultiScale(roi_gray, 1.3, 20)
        # for (sx, sy, sw, sh) in smile:
        #     cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
