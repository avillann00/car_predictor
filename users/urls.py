from django.urls import path
from .views import register, login, logout

urlpatterns = [
    path('login/', login.as_view(), name='login'),
    path('logout/', logout.as_view(), name='logout'),
    path('register/', register.as_view(), name='register'),
]
