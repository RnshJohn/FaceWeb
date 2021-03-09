# -*- coding:utf-8 _*-
""" 
@Author: John
@Email: workspace2johnwu@gmail.com
@License: Apache Licence 
@File: create_image_dir.py 
@Created: 2020/12/15
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
import shutil
import os
import csv


def create_folder():
    count = 1
    fhand = open("main/exist_members.csv")
    reader = csv.reader(fhand)

    for row in reader:
        if not row:
            break

        current_face_dir = "data/data_faces/" + "person_" + str(count)
        if not os.path.exists(current_face_dir):
            os.mkdir(current_face_dir)

        shutil.copy("static/images/"+str(row[1]), current_face_dir)
        count += 1

