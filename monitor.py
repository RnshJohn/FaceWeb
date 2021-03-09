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


res = subprocess.Popen('ps -ef | grep cognition_3', stdout=subprocess.PIPE, shell=True)
attn = res.stdout.readlines()
counts = len(attn)  # 获取ASRS下的进程个数
print(counts)
if counts < 10:  # 当进程不够正常运行的个数时，说明某只程式退出了
    os.system('python3.8 /Users/johnwu/Practice_project/face_project/cognition_3.py')

