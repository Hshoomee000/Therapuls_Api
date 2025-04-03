
from fastapi import FastAPI, File, UploadFile
import cv2
import mediapipe as mp # type: ignore
import numpy as np
import tempfile
import os
import tensorflow as tf # type: ignore
import pickle
import logging
from  engineered_features import Landmark as LM
from  engineered_features import Frame  as fm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.getLogger("tensorflow").setLevel(logging.ERROR)

final_dec={}
value_List=[]
work_out = ""


received_exercise = "" #input flutter variable "string"

class ExerciseRequest(BaseModel):
    exercise_name: str  # تغيير "text" إلى "exercise_name"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # اسمح لجميع المصادر (يمكنك تحديد مصادر معينة بدلاً من "*")
    allow_credentials=True,  # اسمح بإرسال الكوكيز مع الطلبات
    allow_methods=["*"],  # اسمح لجميع الطرق (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # اسمح لجميع الهيدرات
)


with open("pose_classification_XGBoost_model2.pkl", "rb") as file:
    model = pickle.load(file)




mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                     smooth_landmarks=True)



def process_video(video_path):
    
    global work_out
    """Process video to extract pose landmarks."""
    try:
        print(f"🔍 Processing video: {video_path}")

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ ERROR: Failed to open video file")
            return {"error": "Failed to open video file"}

        landmarks_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                frame_landmarks = []
                landmarks = results.pose_landmarks.landmark

                for i, landmark in enumerate(mp_pose.PoseLandmark):
                    lm = LM(landmark.name, landmarks[i].x, landmarks[i].y, landmarks[i].z)
                    frame_landmarks.append(lm)

                landmarks_list.append(frame_landmarks)

        cap.release()

        if not landmarks_list:
            print("❌ ERROR: No landmarks detected")
            return {"error": "No landmarks detected"}

        for lm in landmarks_list:
            try:
                NOSE = lm[0]
                LEFT_EYE_INNER = lm[1]
                LEFT_EYE = lm[2]
                LEFT_EYE_OUTER = lm[3]
                RIGHT_EYE_INNER = lm[4]
                RIGHT_EYE = lm[5]
                RIGHT_EYE_OUTER = lm[6]
                LEFT_EAR = lm[7]
                RIGHT_EAR = lm[8]
                MOUTH_LEFT = lm[9]
                MOUTH_RIGHT = lm[10]
                LEFT_SHOULDER = lm[11]
                RIGHT_SHOULDER = lm[12]
                LEFT_ELBOW = lm[13]
                RIGHT_ELBOW = lm[14]
                LEFT_WRIST = lm[15]
                RIGHT_WRIST = lm[16]
                LEFT_PINKY = lm[17]
                RIGHT_PINKY = lm[18]
                LEFT_INDEX_FINGER = lm[19]
                RIGHT_INDEX_FINGER = lm[20]
                LEFT_THUMB = lm[21]
                RIGHT_THUMB = lm[22]
                LEFT_HIP = lm[23]
                RIGHT_HIP = lm[24]
                LEFT_KNEE = lm[25]
                RIGHT_KNEE = lm[26]
                LEFT_ANKLE = lm[27]
                RIGHT_ANKLE = lm[28]
                LEFT_HEEL = lm[29]
                RIGHT_HEEL = lm[30]
                LEFT_FOOT_INDEX = lm[31]
                RIGHT_FOOT_INDEX = lm[32]

                f = fm(
                    NOSE, LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE,
                    RIGHT_EYE_OUTER, LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT, LEFT_SHOULDER,
                    RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_PINKY,
                    RIGHT_PINKY, LEFT_INDEX_FINGER, RIGHT_INDEX_FINGER, LEFT_THUMB, RIGHT_THUMB,
                    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
                    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
                )

                final_dec = f.return_value()
                f.return_zero()
                

            except IndexError as e:
                print(f"❌ ERROR: Landmark index error - {e}")
                return {"error": "Landmark index error", "details": str(e)}

    except Exception as e:
        print(f"❌ ERROR in process_video: {e}")
        return {"error": "Unexpected error", "details": str(e)}
    return final_dec

# def InCorrect_movments(finalpredection,l):

#     List_of_wronge_frams = []

#     ind = 1
#     # print(first_indexfounder(l,finalpredection))
#     # print(last_indexfounder(l, finalpredection))
#     ind1 = first_indexfounder(l, finalpredection)
#     ind2 = last_indexfounder(l, finalpredection)
#     final_list = l[ind1:ind2]
#     for lan in final_list:
#         if lan != finalpredection:
#             if ind1 is not None:
#                 List_of_wronge_frams.append(ind1 + ind)
#             else:
#                  print("❌ ind1 is None!")
#                  ind += 1

#     if   len(List_of_wronge_frams) != 0:
#         return (List_of_wronge_frams)
#     else:
#         return ("all is well in your",finalpredection)

def InCorrect_movments(finalpredection,l):

    List_of_wronge_frams = []

    ind = 1
    # print(first_indexfounder(l,finalpredection))
    # print(last_indexfounder(l, finalpredection))
    ind1 = first_indexfounder(l, finalpredection)
    ind2 = last_indexfounder(l, finalpredection)
    final_list = l[ind1:ind2]
    # print("final list is",final_list)
    # print("final predectionis", finalpredection)
    for index,dic in enumerate(final_list):
        key = next(iter(dic))
        if key != finalpredection:
            List_of_wronge_frams.append(dic[key])

            # ind += 1
    print(final_list)
    if  len(List_of_wronge_frams)!=0:
        return("incorrect movment detected at the frams ",List_of_wronge_frams)
    else:
        return ("all is well in your",finalpredection)


def first_indexfounder(lst, finalpred):

    predection_index = 0
    for index,dic in enumerate(lst):
        key=next(iter(dic))
        # print("In first_indexfounder and have:", key)
        if key != finalpred:
          # print(" in condition",key,index)
          #   # predection_index += 1
            continue
        else:

            return index+1


def last_indexfounder(lst, finalpred):
    i=0
    predection_index = len(lst)
    for index, dic in enumerate(reversed(lst)):
        key = next(iter(dic))


        if key != finalpred:
                # print(len(lst) - 1 - index,dic)
                continue
        else:
            # print(lst)
            # print("offfffffffffffff",lst[len(lst)  - index])
            return len(lst)-index


@app.post("/analyze_video")
async def analyze_video(video: UploadFile = File(...)):
    """ استقبال فيديو وتحليل الحركات باستخدام النموذج الجاهز """
    
    # حفظ الفيديو في ملف مؤقت
    temp_video_path = os.path.join(tempfile.gettempdir(), video.filename)
    with open(temp_video_path, "wb") as temp_file:
        temp_file.write(await video.read())

    # استدعاء دالة `process_video` وتمرير مسار الفيديو بدلاً من `frame`
    # final_dec=f.return_zero()
    
    movement_values = process_video(temp_video_path)
    if movement_values == "pushups_up":
        movement_values= "Push-up"

    #if str(work_out) == str(movement_values):
        from engineered_features import li  
        answer = InCorrect_movments(movement_values , li)
        print(answer)
        #return ("is correct work out : ", answer)
        return (answer)
    #else: 
    #    return ("is not correct work out !!!")
    
    # "is not cirrect work out!!"+
    # حذف الملف المؤقت بعد الانتهاء
    os.remove(temp_video_path)
    #if work_out == movement_values:
    

    return {"movement": movement_values}  # إرجاع القيم المستخرجة من الفيديو


@app.post("/txt_input")
async def process_exercise(request: ExerciseRequest):

    global work_out

    received_exercise = request.exercise_name
    work_out = received_exercise
    # print(f"Received exercise: {received_exercise}")  # طباعة في الـ Terminal
    return {work_out}