from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse
from app.models import image
# Create your views here.

def home(requests):
    if requests.method=="POST":
        k = image(image_model=requests.FILES)
        k.save()

        return HttpResponse("seccess")
    t = loader.get_template("home.html")
    return  HttpResponse(t.render({},requests))
