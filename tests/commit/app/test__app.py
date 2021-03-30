from unittest import TestCase

import numpy as np

from phi.vis import App, EditableFloat, EditableBool, EditableString, EditableInt


class TestApp(TestCase):

    def test_app(self):
        app = App('test')
        app.add_field("Random Scalar", np.random.rand(1, 16, 16, 1))
        app.add_field("Random Vector", np.random.rand(1, 16, 16, 3))
        app.add_field("Evolving Scalar", lambda: np.random.rand(1, 16, 16, 1))
        app.value_temperature = 39.5
        app.value_windows_open = False
        app.value_message = "It's too hot!"
        app.value_limit = 42
        app.temperature2 = EditableFloat("Temperature", 40.2)
        app.windows_open2 = EditableBool("Windows Open?", False)
        app.message2 = EditableString("Message", "It's too hot!")
        app.limit2 = EditableInt("Limit", 42, (20, 50))
        app.action_click_here = lambda: app.info('Thanks!')
        app.info("Message")
        app.prepare()

