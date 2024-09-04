from django.shortcuts import render ,HttpResponse

# Create your views here.
def index(reques):
    return    HttpResponse('index  page  for the  sentiment ')