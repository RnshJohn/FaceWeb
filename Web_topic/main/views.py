from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .form import RegisterForm, LoginForm, ProfileForm, PwdChangeForm, RegistGroupForm, ComfirmGroupPassword, PostForm
from django.http import HttpResponseRedirect
from .models import Customer, CustomGroup, EmotionList, Post
from django.urls import reverse
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import auth
from django.shortcuts import render, HttpResponse, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import Group
from django.contrib.auth import authenticate, login, logout
from .decorators import unauthenticated_user, allowed_users, admin_only
from django.contrib import messages
from django.contrib.auth.models import Group
import json
import datetime
from collections import Counter
from django.template.defaultfilters import slugify
from taggit.models import Tag
import os
import csv
from scripts import emotion_load, load_database_name




@login_required(login_url='login')
@allowed_users(allowed_roles=['customer'])
def group_password_smallPage(request, pk_name):
    # if request.method == 'GET':
    #     return render(request, 'main/group_password_smallPage.html')
    if request.method == 'POST':
        form = ComfirmGroupPassword(request.POST)
        if form.is_valid():
            password = form.cleaned_data['password']
            customer = request.user.customer

            user = get_object_or_404(User, username=customer)
            group = CustomGroup.objects.get(name=pk_name)

            group_slug = CustomGroup.has_members.get_Slug(name=pk_name)
            if password == group.password:
                print("comfirm password....")
                group.user_set.add(user)
                return HttpResponseRedirect(reverse("main:group-page", args=[group_slug]))
            else:
                messages.error(request, "密碼錯誤")
    form = ComfirmGroupPassword()
    return render(request, 'main/group_password_smallPage.html', {'form': form, 'group_name': pk_name})


def homepage(request):
    print("Homepage")



    return HttpResponse("index ok")

@login_required(login_url='login')
@allowed_users(allowed_roles=['customer'])
def userPage(request):

    emotion_load.run()
    if request.method == 'POST':
        if 'search' in request.POST:

            groupname = request.POST.get('groupname')
            group_comfirm = CustomGroup.objects.filter(name=groupname)
            # group = CustomGroup.objects.filter(name=groupname)

            if not group_comfirm:
                messages.error(request, "沒有此群組")
                return redirect("main:user-page")
            group = CustomGroup.objects.get(name=groupname)
            groupslug = CustomGroup.has_members.get_Slug(name=groupname)
            customer = request.user.customer
            user = group.user_set.filter(username=customer)

            if user:
                return HttpResponseRedirect(reverse("main:group-page", args=[groupslug]))
            else:
                messages.error(request, "您沒有此權限，跳轉頁面輸入密碼")
                return HttpResponseRedirect(reverse('main:group_password_smallPage', args=[groupname]))
    customer = request.user.customer
    user = get_object_or_404(User, username=customer)
    user_profile = get_object_or_404(Customer, user=user)

    return render(request, 'main/Home.html', {'user': user, 'image': user_profile})


@login_required(login_url='login')
@allowed_users(allowed_roles=['customer'])
def addGroup(request):
    if request.method == 'POST':
        if 'registeredGroup' in request.POST:
            form = RegistGroupForm(request.POST, request.FILES)
            if form.is_valid():
                groupname = form.cleaned_data['groupname']
                image = form.cleaned_data['groupImage']
                password = form.cleaned_data['password2']
                group = CustomGroup.objects.filter(name=groupname)
                # comfirm group is exist
                if not group:
                    group = CustomGroup(name=groupname, group_pic=image, password=password)
                    group.save()
                    group = CustomGroup.objects.get(name=groupname)
                    group_slug = CustomGroup.has_members.get_Slug(name=groupname)
                    customer = request.user.customer
                    user = get_object_or_404(User, username=customer)
                    group.user_set.add(user)
                    return HttpResponseRedirect(reverse("main:group-page", args=[group_slug]))
                else:
                    messages.error(request, "已有此群組")
                    return HttpResponseRedirect(reverse("main:add-group"))

        if 'search' in request.POST:
            groupname = request.POST.get('groupname')
            group_comfirm = CustomGroup.objects.filter(name=groupname)
            # group = CustomGroup.objects.filter(name=groupname)

            if not group_comfirm:
                messages.error(request, "沒有此群組")
                return redirect("main:user-page")
            group = CustomGroup.objects.get(name=groupname)
            groupslug = CustomGroup.has_members.get_Slug(name=groupname)
            customer = request.user.customer
            user = group.user_set.filter(username=customer)

            if user:
                return HttpResponseRedirect(reverse("main:group-page", args=[groupslug]))
            else:
                messages.error(request, "您沒有此權限，跳轉頁面輸入密碼")
                return HttpResponseRedirect(reverse('main:group_password_smallPage', args=[groupname]))

    else:
        form = RegistGroupForm()
    customer = request.user.customer
    user = get_object_or_404(User, username=customer)
    user_profile = get_object_or_404(Customer, user=user)
    return render(request, 'main/Add_Group.html', {'form': form, 'user': user, 'image': user_profile})


@login_required(login_url='login')
@allowed_users(allowed_roles=['customer'])
def groupPage(request, pk):
    if 'search' in request.POST:
        groupname = request.POST.get('groupname')
        group_comfirm = CustomGroup.objects.filter(name=groupname)
        # group = CustomGroup.objects.filter(name=groupname)

        if not group_comfirm:
            messages.error(request, "沒有此群組")
            return redirect("main:user-page")
        group = CustomGroup.objects.get(name=groupname)
        groupslug = CustomGroup.has_members.get_Slug(name=groupname)
        customer = request.user.customer
        user = group.user_set.filter(username=customer)

        if user:
            return HttpResponseRedirect(reverse("main:group-page", args=[groupslug]))
        else:
            messages.error(request, "您沒有此權限，跳轉頁面輸入密碼")
            return HttpResponseRedirect(reverse('main:group_password_smallPage', args=[groupname]))

    customer = request.user.customer
    user = get_object_or_404(User, username=customer)
    user_profile = get_object_or_404(Customer, user=user)

    group = CustomGroup.objects.get(slug=pk)
    check_users = group.user_set.all()

    all_customers = []
    customers_pic = []
    for check_user in check_users:
        # check_customer = Customer.user.objects.get(username=check_user)
        check_customer = get_object_or_404(User, username=check_user)
        user_profile1 = get_object_or_404(Customer, user=check_customer)
        all_customers.append(check_customer.username)
        customers_pic.append(user_profile1.image.url)
    nums = len(all_customers)
    print(customers_pic)
    json_customers = json.dumps(all_customers)
    json_pic = json.dumps(customers_pic)

    return render(request, 'main/GroupHome.html', {'user': user, 'image': user_profile, 'all_customers': json_customers,
                                                   'nums': nums, 'customer_pic': json_pic})


@login_required(login_url='login')
@allowed_users(allowed_roles=['customer'])
def emotionPage(request, pk):
    login = request.user.customer
    login_user = get_object_or_404(User, username=login)

    user = get_object_or_404(User, username=pk)
    user_profile = Customer.objects.get(user=user)

    current_time = datetime.datetime.now()

    day_list = []
    weak_list = []

    dayDate = current_time - datetime.timedelta(days=1)
    check_day_datas = EmotionList.objects.filter(customer=user, data_created__range=[dayDate, current_time])
    weakDate = current_time - datetime.timedelta(days=6)
    check_weak_datas = EmotionList.objects.filter(customer=user, data_created__range=[weakDate, current_time])

    if check_day_datas:
        for day_data in check_day_datas:
            day_list.append(day_data.status)

    if check_weak_datas:
        for week_data in check_weak_datas:
            weak_list.append(week_data.status)

    dayDic = Counter(day_list)
    weekDic = Counter(weak_list)

    # trans day dic to list
    day_count_list = [0, 0, 0, 0, 0, 0, 0]
    angry_value = dayDic.get('angry')
    if angry_value:
        day_count_list[0] = angry_value

    disgust_value = dayDic.get('disgust')
    if disgust_value:
        day_count_list[1] = disgust_value

    fear_value = dayDic.get('fear')
    if fear_value:
        day_count_list[2] = fear_value

    happy_value = dayDic.get('happy')
    if happy_value:
        day_count_list[3] = happy_value

    sad_value = dayDic.get('sad')
    if sad_value:
        day_count_list[4] = sad_value

    surprise_value = dayDic.get('surprise')
    if surprise_value:
        day_count_list[5] = surprise_value

    neutral_value = dayDic.get('neutral')
    if neutral_value:
        day_count_list[6] = neutral_value

    day_Json = json.dumps(day_count_list)

    # trans week dic to list
    week_count_list = [0, 0, 0, 0, 0, 0, 0]
    angry_value = weekDic.get('angry')
    if angry_value:
        week_count_list[0] = angry_value

    disgust_value = weekDic.get('disgust')
    if disgust_value:
        week_count_list[1] = disgust_value

    fear_value = weekDic.get('fear')
    if fear_value:
        week_count_list[2] = fear_value

    happy_value = weekDic.get('happy')
    if happy_value:
        week_count_list[3] = happy_value

    sad_value = weekDic.get('sad')
    if sad_value:
        week_count_list[4] = sad_value

    surprise_value = weekDic.get('surprise')
    if surprise_value:
        week_count_list[5] = surprise_value

    neutral_value = weekDic.get('neutral')
    if neutral_value:
        week_count_list[6] = neutral_value

    week_Json = json.dumps(week_count_list)

    # 列出所有標籤
    posts = Post.objects.filter(target_customer=pk)
    comman_tags = Post.tags.most_common()[:4]

    form = PostForm(request.POST)
    if form.is_valid():
        newPost = form.save(commit=False)
        newPost.slug = slugify(newPost.title)
        newPost.create_customer = login_user
        newPost.target_customer = user.username

        newPost.save()
        form.save_m2m()

    return render(request, 'main/Room_List.html', {'day_list': day_Json, 'week_list': week_Json, 'posts': posts,
                                                   'common_tags': comman_tags,
                                                   'form': form, 'pk': pk, })


def detail_view(request, pk, slug):
    post = get_object_or_404(Post, slug=slug)
    context = {
        'post': post,
    }
    render(request, 'main/post_detail.html', context)

@login_required(login_url='login')
@allowed_users(allowed_roles=['customer'])
def tagged(request, pk, slug):

    login = request.user.customer
    login_user = get_object_or_404(User, username=login)

    user = get_object_or_404(User, username=pk)
    user_profile = Customer.objects.get(user=user)

    current_time = datetime.datetime.now()

    day_list = []
    weak_list = []

    dayDate = current_time - datetime.timedelta(days=1)
    check_day_datas = EmotionList.objects.filter(customer=user, data_created__range=[dayDate, current_time])
    weakDate = current_time - datetime.timedelta(days=6)
    check_weak_datas = EmotionList.objects.filter(customer=user, data_created__range=[weakDate, current_time])

    if check_day_datas:
        for day_data in check_day_datas:
            day_list.append(day_data.status)

    if check_weak_datas:
        for week_data in check_weak_datas:
            weak_list.append(week_data.status)

    dayDic = Counter(day_list)
    weekDic = Counter(weak_list)

    # trans day dic to list
    day_count_list = [0, 0, 0, 0, 0, 0, 0]
    angry_value = dayDic.get('angry')
    if angry_value:
        day_count_list[0] = angry_value

    disgust_value = dayDic.get('disgust')
    if disgust_value:
        day_count_list[1] = disgust_value

    fear_value = dayDic.get('fear')
    if fear_value:
        day_count_list[2] = fear_value

    happy_value = dayDic.get('happy')
    if happy_value:
        day_count_list[3] = happy_value

    sad_value = dayDic.get('sad')
    if sad_value:
        day_count_list[4] = sad_value

    surprise_value = dayDic.get('surprise')
    if surprise_value:
        day_count_list[5] = surprise_value

    neutral_value = dayDic.get('neutral')
    if neutral_value:
        day_count_list[6] = neutral_value

    day_Json = json.dumps(day_count_list)

    # trans week dic to list
    week_count_list = [0, 0, 0, 0, 0, 0, 0]
    angry_value = weekDic.get('angry')
    if angry_value:
        week_count_list[0] = angry_value

    disgust_value = weekDic.get('disgust')
    if disgust_value:
        week_count_list[1] = disgust_value

    fear_value = weekDic.get('fear')
    if fear_value:
        week_count_list[2] = fear_value

    happy_value = weekDic.get('happy')
    if happy_value:
        week_count_list[3] = happy_value

    sad_value = weekDic.get('sad')
    if sad_value:
        week_count_list[4] = sad_value

    surprise_value = weekDic.get('surprise')
    if surprise_value:
        week_count_list[5] = surprise_value

    neutral_value = weekDic.get('neutral')
    if neutral_value:
        week_count_list[6] = neutral_value

    week_Json = json.dumps(week_count_list)

    # 列出所有標籤
    posts = Post.objects.filter(target_customer=pk)
    comman_tags = Post.tags.most_common()[:4]

    tag = get_object_or_404(Tag, slug=slug)
    user_post = Post.objects.filter(target_customer=pk)

    posts = user_post.filter(tags=tag)
    common_tags = Post.tags.most_common()[:4]


    context = {
        'tag': tag,
        'common_tags': common_tags,
        'posts': posts,
        'day_list': day_Json, 'week_list': week_Json,
        'pk': pk,
    }
    return render(request, 'main/Room_List.html', context)


@login_required(login_url='login')
@allowed_users(allowed_roles=['customer'])
def profile_update(request):
    customer = request.user.customer

    user = get_object_or_404(User, username=customer)
    user_profile = get_object_or_404(Customer, user=user)
    if request.method == 'POST':
        if 'search' in request.POST:
            groupname = request.POST.get('groupname')
            group_comfirm = CustomGroup.objects.filter(name=groupname)
            # group = CustomGroup.objects.filter(name=groupname)

            if not group_comfirm:
                messages.error(request, "沒有此群組")
                return redirect("main:user-page")
            group = CustomGroup.objects.get(name=groupname)
            groupslug = CustomGroup.has_members.get_Slug(name=groupname)
            customer = request.user.customer
            user = group.user_set.filter(username=customer)

            if user:
                return HttpResponseRedirect(reverse("main:group-page", args=[groupslug]))
            else:
                messages.error(request, "您沒有此權限，跳轉頁面輸入密碼")
                return HttpResponseRedirect(reverse('main:group_password_smallPage', args=[groupname]))

        if "Update_Information" in request.POST:
            form = ProfileForm(request.POST)
            if form.is_valid():
                user.username = form.clean_data['username']
                user.password = form.cleaned_data['password2']
                user.save()

                user_profile.phone_number = form.cleaned_data['phone_number']
                user_profile.save()

                return HttpResponseRedirect(reverse('main:room'))


    else:
        default_data = {
            'username': user.username,

            'phone_number': user_profile.phone_number,
        }
        form = ProfileForm(default_data)

    return render(request, 'main/profile_updata.html', {'form': form, 'user': user, 'image': user_profile})


@unauthenticated_user
def register(request):

    if request.method == 'POST':

        form = RegisterForm(request.POST, request.FILES)

        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password2']
            phone_number = form.cleaned_data['phone_number']
            image = form.cleaned_data['image']

            # path_photo = "../static/images/images/"+username
            # if os.path.isdir(path_photo):
            #     pass
            # else:
            #     os.mkdir(path_photo)
            #
            # image_url = username+'/'+str(image)

            group = Group.objects.get(name='customer')
            user = User.objects.create_user(username=username, password=password, email=email)
            group.user_set.add(user)
            user_profile = Customer(user=user, phone_number=phone_number, image=image)

            user_profile.save()

            return HttpResponseRedirect("/account/login")
    else:
        form = RegisterForm()
    return render(request, 'main/registration.html', {'form': form})


@unauthenticated_user
def loginPage(request):
    if request.method == 'POST':
        load_database_name.run()
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

            user = auth.authenticate(username=username, password=password)

            if user is not None and user.is_active:
                auth.login(request, user)
                return HttpResponseRedirect(reverse('main:user-page'))
            else:
                # 登入失敗
                messages.info(request, "Username OR password is incorrect")
                # return render(request, 'main/login.html', {'form': form, "message": "Wrong password Please Try again"})

    else:
        form = LoginForm()
    return render(request, 'main/login.html', {'form': form})


@login_required()
def logout(request):
    auth.logout(request)
    return redirect('main:login')


@login_required()
def pwd_change(request, pk):
    user = get_object_or_404(User, pk=pk)

    if request.method == "POST":
        form = PwdChangeForm(request.POST)

        if form.is_valid():
            password = form.cleaned_data['old_password']
            username = user.username

            user = auth.authenticate(username=username, password=password)

            if user is not None and user.is_activate:
                new_password = form.cleaned_data['password2']
                user.set_password(new_password)
                user.save()
                return HttpResponseRedirect('/account/login/')

            else:
                return render(request, 'main/pwd_change.html',
                              {'form': form, 'user': user, 'message': 'Old password is wrong Try again'})

    else:
        form = PwdChangeForm()

    return render(request, 'main/pwd_change.html', {'form': form, 'user': user})
