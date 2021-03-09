from django.db import models, connection
from django.urls import reverse
from phonenumber_field.modelfields import PhoneNumberField
from django.contrib.auth.models import User, Group
from django.template.defaultfilters import slugify
from taggit.managers import TaggableManager


# Create your models here.


class Customer(models.Model):
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.CASCADE)
    phone_number = models.CharField('Telephone', max_length=128, blank=True)
    image = models.ImageField('Image', null=True, upload_to='images/', blank=True)

    class Meta:
        verbose_name = 'User profile'

    def __str__(self):
        return "{}".format(self.user.__str__())


class CustomGroupManager(models.Manager):
    def get_Slug(self, name):
        return super(CustomGroupManager, self).get(name=name).slug

    def get_Password(self, name):
        return super(CustomGroupManager, self).get(name=name).password

    def get_query_set(self):
        return super(CustomGroupManager, self).get_queryset()


class CustomGroup(Group):
    objects = models.Manager()

    password = models.CharField(max_length=200)
    slug = models.SlugField(allow_unicode=True, unique=True)

    group_pic = models.ImageField('GroupImage', upload_to='group_images/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now=True)
    has_members = CustomGroupManager()

    class Meta:
        verbose_name_plural = "Customers"
        ordering = ['name']

    def __str__(self):
        return "{}".format(self.name)

    def get_absolute(self):
        return reverse("groups:single", kwargs={"slug": self.slug})

    def save(self, *args, **kwargs):
        self.slug = slugify(self.name)
        super(CustomGroup, self).save(*args, **kwargs)


class EmotionList(models.Model):
    STATUS = (
        ('angry', 'angry'),
        ('disgust', 'disgust'),
        ('fear', 'fear'),
        ('happy', 'happy'),
        ('sad', 'sad'),
        ('surprise', 'surprise'),
        ('neutral', 'neutral'),
    )
    is_deleted = models.BooleanField(default=False)
    customer = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    data_created = models.DateTimeField(null=True)
    status = models.CharField(max_length=200, null=True, choices=STATUS)


class Post(models.Model):
    create_customer = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    title = models.CharField(max_length=250)
    description = models.TextField()
    target_customer = models.CharField(max_length=50, null=True)
    published = models.DateField(auto_now_add=True)
    slug = models.SlugField(unique=True, max_length=100, null=True)
    tags = TaggableManager()

    def __str__(self):
        return self.title
