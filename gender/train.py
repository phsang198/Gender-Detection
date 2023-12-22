import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


# Variables
DATASET_DIR = "D:\OUTSOURCE\PYTHON\GENDER\gender"
TRAINING_DIR = os.path.join(DATASET_DIR, "data")
TESTING_DIR = os.path.join(DATASET_DIR, "testing")
MODEL_PATH = "D:\OUTSOURCE\PYTHON\GENDER\gender\model\model_lbph.yml"
LABEL_ENCODER_PATH = "D:\OUTSOURCE\PYTHON\GENDER\gender\model\label_encoder.pkl"

label_encoder = LabelEncoder()

def create_dataset(directory):
    faces = []
    labels = []

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Liệt kê các thư mục con 
    for subdirectory in os.listdir(directory):
        if not subdirectory.startswith("."):
            subdir_path = os.path.join(directory, subdirectory)
            if os.path.isdir(subdir_path):
                # Lặp qua các hình ảnh trong thư mục con
                for image in os.listdir(subdir_path):
                    if not image.startswith("."):
                        img = cv2.imread(os.path.join(subdir_path, image), cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (100, 100))
                        faces.append(img)
                        labels.append(subdirectory)  # Sử dụng tên thư mục con làm nhãn

    return faces, labels

# Tiền xử lí dữ liệu
def preprocess_data(faces, labels):
    faces = np.array(faces)
    labels = np.array(labels)
    labels = label_encoder.fit_transform(labels)

    return faces, labels

# Đào tạo model
def train_model(faces, labels):
    # Chia dữ liệu thành tập huấn luyện và tập kiểm thử
    x_train, x_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

    # Tạo LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Huấn luyện model dùng model set
    recognizer.train(x_train, y_train)

    # Lưu model
    recognizer.save(MODEL_PATH)
    
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    # Dự đoán cho tập dữ liệu kiểm thử
    y_pred = []
    for x in x_test:
        label, _ = recognizer.predict(x)
        y_pred.append(label)

    # Độ chính xác Model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

# Execution
if __name__ == "__main__":
    train_faces, train_labels = create_dataset(TRAINING_DIR)
    train_faces, train_labels = preprocess_data(train_faces, train_labels)

    train_model(train_faces, train_labels)
