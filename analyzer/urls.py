from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('analyze/', views.analyze, name='analyze'),
    path('api/analyze/', views.analyze_api, name='analyze_api'),
    path('stats/', views.stats, name='stats'),
    path('stats/delete/<int:article_id>/', views.delete_article, name='delete_article'),
    path('stats/delete-all/', views.delete_all_history, name='delete_all_history'),
]

