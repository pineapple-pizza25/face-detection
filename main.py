import cv2
import pathlib
import time

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

face_detected = False

while True:
    _, frame = camera.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors = 10,
        minSize = (400,400) ,
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x,y), (x+width, y+width), (255, 255, 0), 2)

        result, image = camera.read()

        cv2.destroyWindow("video")

        cv2.imshow("image", image)
        cv2.waitKey(1) 
        time.sleep(6)
        cv2.destroyWindow("image")

        faces = []        
    else:
        face_detected = False
        cv2.imshow("video", frame)


    if cv2.waitKey(1) == ord("q"):
        break

#cv2.imshow("faces", frame)
camera.release()
cv2.destroyAllWindows()