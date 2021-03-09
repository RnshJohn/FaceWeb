# -*- coding:utf-8 _*-
""" 
@Author: John
@Email: workspace2johnwu@gmail.com
@License: Apache Licence 
@File: cognition.py 
@Created: 2020/11/30
@site:  
@software: PyCharm 

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃            ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神獸保佑    ┣┓
                ┃　永無BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛ 
"""

import dlib
import numpy as np
import cv2
import pandas as pd
import os
import time
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import csv
import datetime
from scripts import create_image_dir, features_to_csv, load_database_name


detector = dlib.get_frontal_face_detector()


predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')


face_reco_model = dlib.face_recognition_model_v1("data/dlib_face_recognition_resnet_model_v1.dat")


# ----------------emotion data------------------------------------------------------
#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')




class Face_Recognizer:
    def __init__(self):
        self.feature_known_list = []
        self.name_known_list = []

        self.current_frame_face_cnt = 0
        self.current_frame_feature_list = []
        self.current_frame_name_position_list = []
        self.current_frame_name_list = []

        self.current_frame_emotion_position_list = []
        self.current_frame_emotion_list = []
        self.current_time_list = []
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        # Update FPS
        self.fps = 0
        self.frame_start_time = 0
        self.csv_count = 0
    # Get known faces from "features_all.csv"
    def get_face_database(self):
        self.update_img_dir()
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.feature_known_list.append(features_someone_arr)
                self.name_known_list.append("Person_" + str(i + 1))
            print("Faces in Database：", len(self.feature_known_list))
            return 1
        else:
            print('##### Warning #####', '\n')
            print("'features_all.csv' not found!")
            print(
                "Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'",
                '\n')
            print('##### End Warning #####')
            return 0

    # Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now


    def update_img_dir(self):
        if load_database_name.update == True:
            create_image_dir.create_folder()
            features_to_csv.run()
            load_database_name.update = False



    def draw_note(self, img_rd):
        font = cv2.FONT_ITALIC

        cv2.putText(img_rd, "Face Recognizer", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_face_cnt), (20, 140), font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def csv_writer(self):
        if self.csv_count == 500:
            with open("../main/emotions.csv", "a+", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for i in range(self.current_frame_face_cnt):
                    if self.current_frame_name_list[i] == 'unknown':
                        self.csv_count = 499
                        continue
                    else:
                        writer.writerow([self.current_frame_name_list[i], self.current_frame_emotion_list[i], self.current_time_list[i]])
            self.csv_count = 0
        else:
            self.csv_count += 1


    def draw_name_and_emotion(self, img_rd):
        # Write names under rectangle

        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            draw.text(xy=self.current_frame_name_position_list[i], text=self.current_frame_name_list[i])
            draw.text(xy=self.current_frame_emotion_position_list[i], text=self.current_frame_emotion_list[i])

            img_with_name = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        return img_with_name

    #  Show names in chinese
    def show_chinese_name(self):

        if self.current_frame_face_cnt >= 1:
            self.name_known_list[0] = 'D0772389'.encode('utf-8').decode()
            self.name_known_list[1] = 'john851411'.encode('utf-8').decode()
            self.name_known_list[2] = 'johnwu'.encode('utf-8').decode()

    def process(self, stream):

        if self.get_face_database():
            while stream.isOpened():
                print(">>> Frame start")
                flag, img_rd = stream.read()
                faces = detector(img_rd, 0)
                kk = cv2.waitKey(1)
                gray_frame = cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY)
                if kk == ord('q'):
                    break
                else:
                    self.draw_note(img_rd)
                    self.current_frame_feature_list = []
                    self.current_frame_face_cnt = 0
                    self.current_frame_name_position_list = []
                    self.current_frame_name_list = []
                    self.current_time_list = []
                    self.current_frame_emotion_list = []
                    self.current_frame_emotion_position_list = []
                    #  Face detected in current frame
                    if len(faces) != 0:
                        # Compute the face descriptors for faces in current frame
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))

                        # Traversal all the faces in the database
                        for k in range(len(faces)):
                            print(">>>>>> For face", k + 1, " in camera")
                            now = datetime.datetime.today()
                            self.current_time_list.append(now)
                            self.current_frame_name_list.append("unknown")
                            self.current_frame_emotion_list.append("unknown")

                            height = faces[k].bottom() - faces[k].top()
                            width = faces[k].right() - faces[k].left()
                            print(faces[k].right())
                            print(faces[k].bottom())
                            h_mid = int(height / 2)
                            w_mid = int(width / 2)

                            if (faces[k].right() + w_mid) > 1280 or (faces[k].bottom() + h_mid) > 750 or (faces[k].left() - w_mid) < 0 or (faces[k].top() - h_mid) <0:
                                # if(k==1):
                                #
                                font = cv2.FONT_ITALIC
                                cv2.putText(img_rd, "OUT OF RANGE", (20, 300), font, 0.8, (0, 0, 255), 1,
                                            cv2.LINE_AA)

                                print(" Please adjust your position to detect emotion")

                            else:
                                # emotion process
                                roi_gray = gray_frame[faces[k].top(): faces[k].bottom(),
                                           faces[k].left(): faces[k].right()]
                                roi_gray = cv2.resize(roi_gray, (48, 48))
                                img_pixels = image.img_to_array(roi_gray)
                                img_pixels = np.expand_dims(img_pixels, axis=0)
                                img_pixels /= 255

                                prediction = model.predict(img_pixels)
                                print(prediction)
                                max_index = np.argmax(prediction[0])
                                predicted_emotion = self.emotions[max_index]

                                self.current_frame_emotion_list[k] = predicted_emotion



                            self.current_frame_emotion_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 2)]
                            ))
                            self.current_frame_name_position_list.append(tuple(
                                [faces[k].left(),
                                 int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            current_frame_e_distance_list = []
                            for i in range(len(self.feature_known_list)):

                                if str(self.feature_known_list[i][0]) != '0.0':
                                    print("   >>> With person", str(i + 1), ", the e distance: ", end='')
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_feature_list[k],
                                        self.feature_known_list[i])
                                    print(e_distance_tmp)
                                    current_frame_e_distance_list.append(e_distance_tmp)
                                else:
                                    #  person_X
                                    current_frame_e_distance_list.append(999999999)
                            # 6.  Find the one with minimum e distance
                            similar_person_num = current_frame_e_distance_list.index(
                                min(current_frame_e_distance_list))
                            print("   >>> Minimum e distance with ", self.name_known_list[similar_person_num], ": ",
                                  min(current_frame_e_distance_list))

                            if min(current_frame_e_distance_list) < 0.4:
                                self.current_frame_name_list[k] = self.name_known_list[similar_person_num]
                                print(
                                    "   >>> Face recognition result:  " + str(
                                        self.name_known_list[similar_person_num]))
                            else:
                                print("   >>> Face recognition result: Unknown person")

                            for kk, d in enumerate(faces):
                                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                              (0, 255, 255), 2)

                        self.current_frame_face_cnt = len(faces)

                        self.show_chinese_name()
                        self.csv_writer()

                        img_with_name = self.draw_name_and_emotion(img_rd)

                    else:
                        img_with_name = img_rd

                print(">>>>>> Faces in camera now:", self.current_frame_name_list)
                print(">>>>>> Emotions in camera now:", self.current_frame_emotion_list)
                cv2.namedWindow("camera", 1)
                cv2.resizeWindow("camera", 640, 480)
                cv2.imshow("camera", img_with_name)

                # 9. 更新 FPS / Update stream FPS
                self.update_fps()
                print(">>> Frame ends\n\n")

    # OpenCV 调用摄像头并进行 process
    # @background(schedule=1)
    def run(self):
        cap = cv2.VideoCapture(0) #外接攝影機:1 ;內部攝影機: 0
        # cap = cv2.VideoCapture("video.mp4")
        cap.set(3, 480)  # 640x480
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
