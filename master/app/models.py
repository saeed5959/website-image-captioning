from django.db import models

# Create your models here.

class image(models.Model):

    image_model = models.ImageField(upload_to='app.static.images')