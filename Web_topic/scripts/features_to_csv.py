# -*- coding:utf-8 _*-
""" 
@Author: John
@Email: workspace2johnwu@gmail.com
@License: Apache Licence 
@File: features_to_csv.py.py 
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

import os
import dlib
from skimage import io
import csv
import numpy as np

path_photo = "data/data_faces/"

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

face_recognition_model = dlib.face_recognition_model_v1("data/dlib_face_recognition_resnet_model_v1.dat")


def return_128d_features(path_img):
    """
    image 128d features return
    :param path_img: <class 'str'> input
    :return: <class 'dlib.vector'>
    """

    img_rd = io.imread(path_img)
    faces = detector(img_rd, 1)

    print("%-40s %-20s" % (" Image with faces detected:", path_img), '\n')

    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_recognition_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        print("no face")
    return face_descriptor


def return_features_mean_personX(path_faces_personX):
    print(path_faces_personX)
    features_list_personX = []
    photos_list = os.listdir(path_faces_personX)
    if photos_list:
        for i in range(len(photos_list)):
            #  Get 128D features for single image of personX
            print("%-40s %-20s" % ("Reading image:", path_faces_personX + "/" + photos_list[i]))
            features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
             # Jump if no face detected from image
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        print("Warning: No images in " + path_faces_personX + '/', '\n')

    # Compute the mean
    # personX 的 N 张图像 x 128D -> 1 x 128D
    print(features_list_personX)
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=int, order='C')
    print(type(features_mean_personX))
    return features_mean_personX


# 获取已录入的最后一个人脸序号 / Get the order of latest person

def run():
    person_list = os.listdir("data/data_faces")
    person_num_list = []
    print(person_list)
    for person in person_list:
        person_num_list.append(int(person.split('_')[-1]))
    person_cnt = max(person_num_list)

    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in range(person_cnt):
            # Get the mean/average features of face/personX, it will be a list with a length of 128D
            print(path_photo + "person_" + str(person + 1))
            features_mean_personX = return_features_mean_personX(path_photo + "person_" + str(person + 1))
            writer.writerow(features_mean_personX)
            print("The mean of features:", list(features_mean_personX))
            print('\n')
        print("Save all the features of faces registered into: data/features_all.csv")
