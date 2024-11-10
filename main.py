import cv2
import pathlib
import time
import requests
import io
import PIL.Image as Image
import numpy as np

api_url = 'https://facial-recognition-api.calmwave-03f9df68.southafricanorth.azurecontainerapps.io/facialrecognition'

clf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

face_detected = False

while True:
    _, frame = camera.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors = 10,
        minSize = (370,370) ,
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        #time.sleep(1)
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x,y), (x+width, y+width), (255, 255, 0), 2)

        result, image = camera.read()

        cv2.destroyWindow("video")

        cv2.imshow("image", image)
        cv2.waitKey(1) 
        time.sleep(6)
        cv2.destroyWindow("image")

        try:
            #image_bytes = io.BytesIO(image)  

            img_encode = cv2.imencode('.png', image)[1] 

            data_encode = np.array(img_encode)
            data = {"array": data_encode.tolist()}

            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = io.BytesIO(buffer)
            file = {'file': ('image.jpg', image_bytes, 'image/jpeg')}

            #files = {"file": open("C:/Users/snmis/OneDrive/Pictures/Camera Roll/WIN_20240903_14_52_19_Pro.jpg", "rb")}

            try:
                response = requests.post(api_url, files=file)
                print('Status Code:', response.status_code)
                print('Response Data:', response.json())  
                
            except requests.exceptions.RequestException as e:
                print('Error:', e)
                #return None
            


            print('Success!')

        except requests.exceptions.RequestException as e:
            print(f'An error occurred: {e}')

        faces = []        
    else:
        face_detected = False
        cv2.imshow("video", frame)


    if cv2.waitKey(1) == ord("q"):
        break

#cv2.imshow("faces", frame)
camera.release()
cv2.destroyAllWindows()