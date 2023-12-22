import cv2
import os
from gender.test import predict 
from threading import Thread

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
output_folder = 'output_faces'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

gx = 0
gy = 0 
curr_frame = None

def process_faces(frame, faces, output_folder):
    global gx, gy, curr_frame

    curr_frame = frame 

    count = -1 

    for (x, y, w, h) in faces:
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_roi = frame[y:y+h, x:x+w]
        face_filename = os.path.join(output_folder, f'face_{count}.jpg')
        cv2.imwrite(face_filename, face_roi)

        # Sử dụng luồng để dự đoán giới tính
        gx = x
        gy = y
        predict_thread = Thread(target=process_gender, args=(face_filename,))
        predict_thread.start()

def process_gender(face_filename):
    global gx, gy, curr_frame
    res = predict(face_filename)
    print(res+'\n')
    cv2.putText(curr_frame, res, (gx, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def detect(video_path):
    cap = cv2.VideoCapture(video_path)  # Hoặc sử dụng 0 để sử dụng webcam
    
    while True:
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        process_faces(frame, faces, output_folder)

        cv2.imshow('Detection From Video', frame)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
