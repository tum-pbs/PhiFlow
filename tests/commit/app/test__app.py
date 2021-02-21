from unittest import TestCase

from phi import app


class TestApp(TestCase):

    def test_app_creation(self):
        app_ = app.App('test')
