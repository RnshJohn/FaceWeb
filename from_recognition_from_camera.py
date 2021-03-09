# -*- coding:utf-8 _*-
""" 
@Author: John
@Email: workspace2johnwu@gmail.com
@License: Apache Licence 
@File: from_recognition_from_camera.py 
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
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import pandas as pd
import os
import time

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:

    def __int__(self):

        self.feature_known_list = []  # 用来存放所有录入人脸特征的数组 / Save the features of faces in the database
        self.name_known_list = []  # 存储录入人脸名字 / Save the name of faces in the database

        self.current_frame_face_count = 0  # 存储当前摄像头中捕获到的人脸数 / Counter for faces in current frame
        self.current_frame_feature_list = []  # 存储当前摄像头中捕获到的人脸特征 / Features of faces in current frame
        self.current_frame_name_position_list = []  # 存储当前摄像头中捕获到的所有人脸的名字坐标 / Positions of faces in current frame
        self.current_frame_name_list = []  # 存储当前摄像头中捕获到的所有人脸的名字 / Names of faces in current frame

        # Update FPS
        self.fps = 0
        self.frame_start_time = 0

    def get_face_data(self):
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

    @staticmethod
    def return_eu_distance(feature_1, feature_2):
        feature_1 = np.append(feature_1)
        feature_2 = np.append(feature_2)

        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, frame_rd):
        font = cv2.FONT_ITALIC

        cv2.putText(frame_rd, "Face Recognizer", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_rd, "Faces: " + str(self.current_frame_face_count), (20, 140), font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(frame_rd, "Q: Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, frame_rd):

        font = ImageFont.truetype("simsun.ttc", 30)
        img = Image.fromarray(cv2.cvtColor(frame_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_count):

            draw.text(xy=self.current_frame_name_position_list[i], text=self.current_frame_name_list[i], font=font)
            img_with_name = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_with_name

    def show_full_name(self):
        if self.current_frame_face_count >= 1:
            self.name_known_list[0] = 'john851411'.encode('utf-8').decode()
            self.name_known_list[1] = 'D0772389'.encode('utf-8').decode()

    def process(self, stream):

        if self.get_face_data():
            while stream.isOpened():
                print("Frame Start")
                flag, frame_rd = stream.read()
                faces = detector(frame_rd, 0)
                key_button = cv2.waitKey(1)

                if key_button == ord('q'):
                    break
                else:
                    self.draw_note(frame_rd)

                    self.current_frame_name_list = []
                    self.current_frame_face_count = []
                    self.current_frame_name_position_list = []
                    self.feature_known_list = []

                    # DETECT FACE TRUE
                    if len(faces) != 0:
                        for i in range(len(faces)):
                            shape = predictor(frame_rd, faces[i])
                            self.current_frame_feature_list.append(face_recognition_model.compute_face_descriptor(frame_rd))

                        for k in range(len(faces)):
                            print(">>>For face", k + 1, "in camera")
                            self.current_frame_name_list.append("unknown")

                            self.current_frame_name_position_list.append(
                                tuple([faces[k].left(), int(faces[k].buttom() - faces[k].top()) / 4]))

                            current_frame_e_distance = []

                            for i in range(len(self.feature_known_list)):
                                if str(self.feature_known_list[i][0]) != '0.0':
                                    print("   >>> With person", str(i + 1), ", the e distance: ", end='')
                                    e_distance_temp = self.return_eu_distance(self.current_frame_feature_list[k],
                                                                          self.feature_known_list[i])

                                    print(e_distance_temp)
                                    current_frame_e_distance.append(e_distance_temp)

                                else:
                                    current_frame_e_distance.append(99999999999999)

                            similar_person_num = current_frame_e_distance.index(min(current_frame_e_distance))
                            print("   >>> Minimum e distance with ", self.name_known_list[similar_person_num], ": ",
                                min(current_frame_e_distance))

                            if min(current_frame_e_distance) < 0.4:
                                self.current_frame_name_list[k] = self.name_known_list[similar_person_num]
                                print("   >>> Face recognition result:  " + str(self.name_known_list[similar_person_num]))
                            else:
                                print("   >>> Face recognition result: Unknown person")

                            # 矩形框 / Draw rectangle
                            for kk, d in enumerate(faces):
                            # 绘制矩形框
                                cv2.rectangle(frame_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                          (0, 255, 255), 2)

                        self.current_frame_face_count = len(faces)

                        # 7. 在这里更改显示的人名 / Modify name if needed
                        # self.show_chinese_name()

                        # 8. 写名字 / Draw name
                        img_with_name = self.draw_name(frame_rd)

                    else:
                        img_with_name = frame_rd

                print(">>>>>> Faces in camera now:", self.current_frame_name_list)

                cv2.imshow("camera", img_with_name)

                #  Update stream FPS
                self.update_fps()
                print(">>> Frame ends\n\n")

    def run(self):
        capture = cv2.VideoCapture(1)
        capture.set(3, 480)  # 640x480
        self.process(capture)
        capture.release()
        cv2.destroyAllWindows()


def main():
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
