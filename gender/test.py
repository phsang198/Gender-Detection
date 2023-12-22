
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

model_path = "D:\OUTSOURCE\PYTHON\GENDER\gender\model\model_lbph.yml"
label_path = "D:\OUTSOURCE\PYTHON\GENDER\gender\model\label_encoder.pkl"

def predict(fileName) :

        # Load mô hình từ file đã lưu
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)

        # Đọc ảnh và chuyển đổi sang ảnh đen-trắng
        img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

        # Resize ảnh để phù hợp với kích thước đã sử dụng trong quá trình huấn luyện
        img = cv2.resize(img, (100, 100))

        # Chuyển đổi ảnh thành numpy array
        img = np.array(img)

        # Đưa ảnh vào mô hình để dự đoán
        label, confidence = recognizer.predict(img)

        # Đọc label_encoder từ file đã lưu
        label_encoder = joblib.load(label_path)

        # Chuyển ngược label thành giới tính
        predicted_gender = label_encoder.inverse_transform([label])[0]

        # In kết quả  
        #print("Predicted Gender:", predicted_gender)
        #print("Confidence Level: {:.2f}%".format(confidence))
        #return "Giới tính là : " + predicted_gender
        return predicted_gender
