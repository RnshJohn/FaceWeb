# -*- coding:utf-8 _*-
""" 
@Author: John
@Email: workspace2johnwu@gmail.com
@License: Apache Licence 
@File: emotion_load.py.py 
@Created: 2020/12/09
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
import csv
from main.models import EmotionList, Customer
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
import datetime
def run():

    fhand = open("main/emotions.csv")
    reader = csv.reader(fhand)

    EmotionList.objects.all().delete()

    for row in reader:
        print(row[0])
        if not row:
            break
        user_comfirm = User.objects.filter(username=row[0]).exists()
        print(user_comfirm)
        if not user_comfirm:
            continue

        user = get_object_or_404(User, username=row[0])
        EmotionList.objects.get_or_create(customer=user, status=row[1], data_created=row[2], is_deleted=False)

