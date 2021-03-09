##網站部分

>dijango移植環境(python 3.7)
>
創立dijango專案
----------------------
####MacOS CMD:
    (myenv) ~/djangourls$ django-admin.py startproject Web_topic
    (myenv) ~/djangourls$ cd ./Web_topic
    (myenv) ~/Web_topic$ python manage.py startapp main
    
####將所有Web_topic底下資料全部複製到你所創立專案的地方（manage.py除外），並執行
    (myenv) ~/Web_topic$ pip3 install -r requirments.txt
    
    
####安裝好所需套件後
    (myenv) ~/Web_topic$ python manage.py makemigrations
    (myenv) ~/Web_topic$ python manage.py migrates
    (myenv) ~/Web_topic$ python manage.py runserver
    
####開啟網頁，並輸入127.0.0.1:8000/account/login
   
   

##人臉辨識部分
>所需套件dlib,tensorflow請自行安裝
>
>開啟Web_topic/scripts位置，執行cognition3.py
>
>情緒儲存位置為Web_topic/main/emotions.csv
    

