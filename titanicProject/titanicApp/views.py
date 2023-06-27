from django.shortcuts import render
from joblib import load

model = load('./savedmodels/model.joblib')

def predictor(request):
    return render(request, 'index.html')

def formInfo(request):
    pclass = request.POST['pclass']
    age = request.POST['age']
    sex = request.POST['sex']
    fare = request.POST['fare']
    y_pred = model.predict([[pclass,age,fare,sex]])
    if y_pred[0] == 0:
        y_pred = "did not survive"
    else:
        y_pred = "Survived"
    return render(request, 'result.html',{'result': y_pred})
