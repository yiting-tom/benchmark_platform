from django.test import TestCase, Client
from django.contrib.auth.models import User
from competitions.models import RegistrationWhitelist

class RegistrationTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.register_url = "/register/"
        self.whitelist_username = "allowed_user"
        RegistrationWhitelist.objects.create(username=self.whitelist_username)

    def test_registration_with_whitelist(self):
        """Test that a whitelisted user can register."""
        response = self.client.post(self.register_url, {
            "username": self.whitelist_username,
            "email": "allowed@example.com",
            "password1": "ComplexPass123!",
            "password2": "ComplexPass123!",
        })
        # Check for redirect to login
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(username=self.whitelist_username).exists())

    def test_registration_without_whitelist(self):
        """Test that a non-whitelisted user cannot register."""
        forbidden_username = "forbidden_user"
        response = self.client.post(self.register_url, {
            "username": forbidden_username,
            "email": "forbidden@example.com",
            "password1": "ComplexPass123!",
            "password2": "ComplexPass123!",
        })
        # Should stay on registration page with error
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "This username is not in the registration whitelist.")
        self.assertFalse(User.objects.filter(username=forbidden_username).exists())

    def test_registration_duplicate_user(self):
        """Test that an existing user cannot register again."""
        User.objects.create_user(username=self.whitelist_username, password="ComplexPass123!")
        response = self.client.post(self.register_url, {
            "username": self.whitelist_username,
            "email": "allowed@example.com",
            "password1": "ComplexPass123!",
            "password2": "ComplexPass123!",
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "A user with that username already exists.")
