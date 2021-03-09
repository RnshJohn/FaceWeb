# -*- coding:utf-8 _*-
""" 
@Author: John
@Email: workspace2johnwu@gmail.com
@License: Apache Licence 
@File: get_face_data.py.py 
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
import os
import shutil
import time
from subprocess import check_call

detector = dlib.get_frontal_face_detector()


class Face_register_in_dir:
    def __init__(self):
        self.path_photos = "data/data_faces/"
        self.font = cv2.FONT_ITALIC

        self.existing_faces_count = 0
        self.unknown_faces_count = 0
        self.current_frame_faces_count = 0

        self.save_flag = 1
        self.press_n_flag = 0

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

    # init the photo data dir
    def init_mkdir(self):

        if os.path.isdir(self.path_photos):
            pass
        else:
            os.mkdir(self.path_photos)

    # delete old data of faces
    def init_del_old_face_data(self):
        folder_rd = os.listdir(self.path_photos)

        for i in range(len(folder_rd)):
            shutil.rmtree(self.path_photos + folder_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")

    # check exist faces in folder
    def check_existing_faces(self):
        if os.listdir(self.path_photos):
            person_list = os.listdir(self.path_photos)
            person_num_list = []

            for person in person_list:
                person_num_list.append(int(person.split('_')[-1]))

            self.existing_faces_count = max(person_num_list)

        else:
            self.existing_faces_count = 0

    def process_after_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, frame_rd):
        cv2.putText(frame_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(frame_rd, "Faces: " + str(self.current_frame_faces_count), (20, 140), self.font, 0.8, (0, 255, 0),
                    1,
                    cv2.LINE_AA)
        cv2.putText(frame_rd, "N: Create face folder", (20, 350), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_rd, "S: Save current face", (20, 400), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def process_faces(self, stream):
        self.init_mkdir()

        if os.path.isdir(self.path_photos):
            self.init_del_old_face_data()

        self.check_existing_faces()

        while stream.isOpened():
            flag, frame_rd = stream.read()
            key_button = cv2.waitKey(1)
            # user dlib detector
            faces = detector(frame_rd, 0)

            if key_button == ord('n'):
                self.existing_faces_count += 1
                current_face_dir = self.path_photos + "person_" + str(self.existing_faces_count)
                os.mkdir(current_face_dir)
                print('\n')
                print("Create folders:", current_face_dir)

                self.unknown_faces_count = 0
                self.press_n_flag = 1

            if len(faces) != 0:
                for k, d in enumerate(faces):
                    x1 = d.left()
                    y1 = d.top()
                    x2 = d.right()
                    y2 = d.bottom()

                    height = y2 - y1
                    width = x2 - x1

                    h_mid = int(height / 2)
                    w_mid = int(width / 2)

                    if (x2 + w_mid) > 640 or (y2 + h_mid > 480) or (x1 - w_mid < 0) or (y1 - h_mid < 0):
                        cv2.putText(frame_rd, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                        color_rectangle = (0, 0, 255)
                        save_flag = 0
                        if key_button == ord('s'):
                            print(" Please adjust your position")
                    else:
                        color_rectangle = (255, 255, 255)
                        save_flag = 1

                    cv2.rectangle(frame_rd,
                                  tuple([x1 - w_mid, y1 - h_mid]),
                                  tuple([x2 + w_mid, y2 + h_mid]),
                                  color_rectangle, 2)

                    img_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)

                    if save_flag:

                        if key_button == ord('s'):

                            if self.press_n_flag:

                                self.unknown_faces_count += 1
                                for ii in range(height * 2):
                                    for jj in range(width * 2):
                                        img_blank[ii][jj] = frame_rd[y1 - h_mid + ii][x1 - w_mid + jj]
                                    img_blank = cv2.resize(img_blank, (64, 64))
                                cv2.imwrite(current_face_dir + "/img_face_" + str(self.unknown_faces_count) + ".jpg", img_blank)
                                print("Save into：",
                                      str(current_face_dir) + "/img_face_" + str(self.unknown_faces_count) + ".jpg")
                            else:
                                print("Please press 'N' to create folder and press 'S'")

                self.current_frame_faces_cnt = len(faces)

                # Add note on cv2 window
                self.draw_note(frame_rd)

                # s Press 'q' to exit
                if key_button == ord('q'):
                    break

                # 11. Update FPS
                self.process_after_fps()

                cv2.namedWindow("camera", 1)
                cv2.resizeWindow("camera", 640, 480)
                cv2.imshow("camera", frame_rd)

    def run_camera(self):
        capture = cv2.VideoCapture(0)
        self.process_faces(capture)

        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Face_register_tem = Face_register_in_dir()
    Face_register_tem.run_camera()
