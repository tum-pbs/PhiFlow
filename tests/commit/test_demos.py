import sys
from os.path import join, dirname, abspath
from unittest import TestCase

import phi
import phi.vis._vis_base as display
from phi.field import Field
from phi.math import backend

DEMOS_DIR = join(dirname(dirname(dirname(abspath(__file__)))), 'demos')
BACKENDS = list(phi.detect_backends())
BACKENDS = tuple([b for b in BACKENDS if b.name != 'Jax'])


def validate_fields(app):
    for name in app.field_names:
        value = app.get_field(name)
        assert isinstance(value, Field) or value is None or backend.choose_backend(value, raise_error=False) is not None, \
            f"Field '{name}' has an invalid value: {value}"


class PerformModelTests(display.Gui):

    def __init__(self):
        display.Gui.__init__(self, asynchronous=False)

    def setup(self, app):
        display.Gui.setup(self, app)
        app.prepare()
        validate_fields(app)
        app.progress()
        app.progress()
        validate_fields(app)
        print("Tests successful.")
        raise InterruptedError

    def auto_play(self):
        pass


def demo_run(name, backends=BACKENDS):
    if DEMOS_DIR not in sys.path:
        print(f"Registering Python source directory {DEMOS_DIR}")
        sys.path.append(DEMOS_DIR)
    with display.force_use_gui(PerformModelTests()):
        for backend_ in backends:
            with backend_:
                print(f"Testing demo {name}.py with {backend_}")
                try:
                    __import__(name)
                except InterruptedError:
                    print(f'Test {name} successfully interrupted.')  # the demos are interrupted after a few steps


class TestDemos(TestCase):

    def test_burgers_sim(self):
        demo_run('burgers_sim')

    # def test_differentiate_pressure(self):
    #     demo_run('differentiate_pressure', [b for b in BACKENDS if b.supports(Backend.gradients)])

    def test_flip_liquid(self):
        demo_run('flip_liquid', [backend.NUMPY_BACKEND])

    def test_fluid_logo(self):
        demo_run('fluid_logo', [backend.NUMPY_BACKEND])  # TODO error with PyTorch on Python 3.8

    def test_heat_equilibrium(self):
        demo_run('heat_equilibrium')

    def test_hw2d(self):
        demo_run('hw2d')

    def test_marker(self):
        demo_run('marker')

    # def test_network_training_pytorch(self):  # TODO Double/Float error on GitHub Actions
    #     demo_run('network_training_pytorch', [b for b in BACKENDS if b.name == 'PyTorch'])

    def test_pipe(self):
        demo_run('pipe')

    def test_point_cloud(self):
        demo_run('point_cloud')

    def test_profile_navier_stokes(self):
        demo_run('profile_navier_stokes')

    # def test_rotating_bar(self):
    #     demo_run('rotating_bar')

    def test_smoke_plume(self):
        demo_run('smoke_plume')
