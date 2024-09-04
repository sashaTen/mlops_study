from django.shortcuts import render ,HttpResponse
import joblib
# Create your views here.
def index(request):
    return   render( request , 'home.html')



def  sentiment(request): 
    loaded_model = joblib.load('sentiment_model.pkl')
    loaded_vectorizer = joblib.load('vectorizer.pkl')
    sentiment = request.POST['sentiment']
    sentiment =  [sentiment]
    sentiment_vectorized = loaded_vectorizer.transform(sentiment)

# Make a prediction
    prediction = loaded_model.predict(sentiment_vectorized)

# Output the prediction
    return HttpResponse(f'Predicted sentiment: {prediction[0]}')


