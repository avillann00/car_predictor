# import required libraries
from allauth.account.views import SignupView, LoginView, LogoutView

# change the default allauth templates to my own

class register(SignupView):
    template_name = 'account/register.html'

class login(LoginView):
    template_name = 'account/login.html'

class logout(LogoutView):
    template_name = 'account/logout.html'
