from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('analyze/', views.analyze, name='analyze'),
    path('api/analyze/', views.analyze_api, name='analyze_api'),
    path('stats/', views.stats, name='stats'),
]

