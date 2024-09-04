from django.shortcuts import render ,HttpResponse

# Create your views here.
def index(request):
    return   render( request , 'home.html')



def  sentiment(request): 
    sentiment = request.POST['sentiment']
    return HttpResponse(sentiment)