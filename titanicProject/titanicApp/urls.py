from . import views
from django.urls import path

urlpatterns = [
    path('', views.predictor, name="predictor"),
    path('result',views.formInfo, name="result"),
]