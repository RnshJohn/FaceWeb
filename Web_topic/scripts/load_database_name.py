import csv
from main.models import Customer
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
import os
import shutil


update = False

def run():
    global update
    update = True
    all_user = User.objects.all()

    with open("main/exist_members.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for user in all_user:
            if user.username == 'rnsh851411':
                continue
            user = get_object_or_404(User,username=user)
            user_profile = get_object_or_404(Customer, user=user)

            writer.writerow([user.username, user_profile.image])




