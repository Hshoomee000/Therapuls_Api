# from fastapi import FastAPI
# from pydantic import BaseModel
# app = FastAPI()

# class Student (BaseModel):
#     id: int
#     name: str
#     grade: int


# Students = [
#     Student(id=1 , name='hesham',grade=80),
#     Student(id=2,name="hosam", grade=70)
# ]

# @app.get("/Students")
# def read_root():
#    return Students

# #@app.get("/")
# #def read_root():
# #    return "hello"

# @app.post("/Students")
# def create_student(New_student : Student):
#     Students.append(New_student)
#     return New_student


# @app.put("/Students{id_student}")
# def update_student(student_id:int , updated_student : Student):
#     for index , Student in enumerate (Students):
#         if Student.id == student_id:
#             Students[index]  = updated_student
#             return update_student 
#     return {'error': "student not found "} 


# @app.delete("/Students{id_student}")
# def delet_student(student_id : int ):
#     for index,Student in enumerate(Students):
#         if Student.id == student_id:
#             del Students[index]
#             return {"massage":"Student deleted"}
#     return{"error":"Student not found"}

from fastapi import FastAPI, File, UploadFile
import cv2
import mediapipe as mp # type: ignore
import numpy as np
import tempfile
import os
import tensorflow as tf # type: ignore
import pickle
import logging
from fastapi.middleware.cors import CORSMiddleware
logging.getLogger("tensorflow").setLevel(logging.ERROR)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # اسمح لجميع المصادر (يمكنك تحديد مصادر معينة بدلاً من "*")
    allow_credentials=True,  # اسمح بإرسال الكوكيز مع الطلبات
    allow_methods=["*"],  # اسمح لجميع الطرق (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # اسمح لجميع الهيدرات
)


with open("pose_Fast_classification_XGBoost_model.pkl", "rb") as file:
    model = pickle.load(file)




mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                     smooth_landmarks=True)


# تحميل الموديل الجاهز لتصنيف الحركات
# MODEL_PATH = "pose_Fast_classification_XGBoost_model.pkl"  # ضع مسار الموديل هنا
# model = tf.keras.models.load_model(MODEL_PATH)

# def extract_pose_angles(frame):
#     """ استخراج الزوايا المطلوبة باستخدام MediaPipe Pose """
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)
#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark

#         def get_angle(a, b, c):
#             a = np.array(a)
#             b = np.array(b)
#             c = np.array(c)
#             ba = a - b
#             bc = c - b
#             cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#             angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#             return np.degrees(angle)

#         keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 
#                      9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
#                      20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        
#         angles = []
#         landmark_points = {i: (landmarks[i].x, landmarks[i].y) for i in keypoints}

#         for i in keypoints:
#             for j in keypoints:
#                 for k in keypoints:
#                     if i < j < k:
#                         angles.append(get_angle(landmark_points[i], landmark_points[j], landmark_points[k]))
        
#         return angles
#     return []
def extract_pose_angles(frame):
    """ استخراج الزوايا المطلوبة باستخدام MediaPipe Pose """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        def get_angle(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)

        keypoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # مفاصل رئيسية فقط
        angle_pairs = [
            (11, 13, 15), (12, 14, 16), (13, 15, 23), (14, 16, 24),
            (23, 25, 27), (24, 26, 28), (11, 23, 25), (12, 24, 26)
        ]

        angles = []
        for a, b, c in angle_pairs:
            angles.append(get_angle(
                (landmarks[a].x, landmarks[a].y),
                (landmarks[b].x, landmarks[b].y),
                (landmarks[c].x, landmarks[c].y),
            ))

        # التأكد من أن الطول دائمًا 99
        while len(angles) < 99:
            angles.append(0)

        return angles

    return [0] * 99  # إرجاع 99 صفرًا إذا لم يتم التعرف على أي حركة

@app.post("/analyze_video")
async def analyze_video(video: UploadFile = File(...)):
    """ استقبال فيديو وتحليل الحركات باستخدام النموذج الجاهز """
    temp_video_path = os.path.join(tempfile.gettempdir(), video.filename)
    with open(temp_video_path, "wb") as temp_file:
        temp_file.write(await video.read())

    cap = cv2.VideoCapture(temp_video_path)
    angles_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        angles = extract_pose_angles(frame)
        if angles:
            angles_list.append(angles)

    cap.release()
    os.remove(temp_video_path)

    movement_name = classify_movement(angles_list)

    return {"movement": movement_name}

def classify_movement(angles_list):
    """ استخدام الموديل الجاهز لتصنيف الحركة """
    if not angles_list:
        return "Unknown"
    
    input_data = np.mean(angles_list, axis=0)  # حساب المتوسط لكل زاوية
    input_data = np.expand_dims(input_data, axis=0)  # تحويلها إلى شكل مناسب للموديل
    prediction = model.predict(input_data)
    movement_label = np.argmax(prediction)  # استخراج التصنيف
    
    return f"Movement Type: {movement_label}"

