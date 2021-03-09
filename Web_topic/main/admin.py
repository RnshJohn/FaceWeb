from django.contrib import admin
from .models import Customer, CustomGroup, EmotionList, Post
# Register your models here.


myModels = [Customer, CustomGroup, EmotionList, Post]
admin.site.register(myModels)




