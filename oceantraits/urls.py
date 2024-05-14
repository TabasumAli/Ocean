# urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('evalutepg1/', views.evalutepg1, name='evalutepg1'),
    path('private/', views.private_view, name='private_view'),
    path('public/', views.public_view, name='public_view'),
    path('scrape_and_evaluate/', views.scrape_and_evaluate, name='scrape_and_evaluate'),  # Add this line
]
