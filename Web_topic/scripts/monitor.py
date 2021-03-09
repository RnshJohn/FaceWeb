#-*- coding:utf-8 _*-  
""" 
@Author: John
@Email: workspace2johnwu@gmail.com
@License: Apache Licence 
@File: monitor.py 
@Created: 2020/12/18
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
import subprocess

while True:
    res = subprocess.Popen('ps -ef | grep cognition_3', stdout=subprocess.PIPE, shell=True)
    attn = res.stdout.readlines()
    counts = len(attn)
    print(counts)
    if counts == 0:

        os.system('python3.8 /Users/johnwu/Practice_project/face_project/Web_topic/manage.py runscript script.cognition.py ')


