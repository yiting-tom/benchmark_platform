from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import RegistrationWhitelist

class UserRegistrationForm(UserCreationForm):
    """
    Custom user registration form with username whitelist validation.
    """

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")

    def clean_username(self):
        username = self.cleaned_data.get("username")
        if not RegistrationWhitelist.objects.filter(username=username).exists():
            raise forms.ValidationError(
                "This username is not in the registration whitelist. "
                "Please contact the administrator."
            )
        return username
