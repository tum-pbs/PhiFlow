from unittest import TestCase

import sys
from os.path import join, dirname, abspath
import numpy as np

import phi.app._display as display
from phi.app import ModuleViewer
from phi.field import Field


DEMOS_DIR = join(dirname(dirname(dirname(abspath(__file__)))), 'demos')


class PerformModelTests(display.AppDisplay):

    def setup(self):
        print('Testing model %s...' % self.app.__class__.__name__)
        self.app.prepare()
        print('Model prepared.')
        self.validate_fields()
        superclasses = [b.__name__ for b in self.app.__class__.__bases__]
        if 'App' in superclasses:
            self.app.play(2)
            print('Steps succeeded.')
            self.validate_fields()
            if isinstance(self.app, ModuleViewer):
                self.app.interrupt()
                print('Interrupting loop')
        else:
            print('Skipping steps')
        raise InterruptedError()

    def validate_fields(self):
        for name in self.app.fieldnames:
            value = self.app.get_field(name)
            assert isinstance(value, (np.ndarray, Field)) or value is None, 'Field "%s" has an invalid value: %s' % (name, value)
        print('All fields are valid.')

    def play(self):
        pass

    def show(self, caller_is_main: bool) -> bool:
        print("Not showing")
        return False


def demo_run(name):
    print(f"Testing demo {name}.py")
    if DEMOS_DIR not in sys.path:
        print(f"Registering Python source directory {DEMOS_DIR}")
        sys.path.append(DEMOS_DIR)
    display.DEFAULT_DISPLAY_CLASS = PerformModelTests
    display.KEEP_ALIVE = False
    try:
        __import__(name)
    except InterruptedError:
        print(f'Test {name} successfully interrupted.')  # the demos are interrupted after a few steps


class TestDemos(TestCase):

    def test_burgers_sim(self):
        demo_run('burgers_sim')

    def test_differentiate_pressure(self):
        demo_run('differentiate_pressure')

    def test_flip_liquid(self):
        demo_run('flip_liquid')

    def test_fluid_logo(self):
        demo_run('fluid_logo')

    def test_heat_equilibrium(self):
        demo_run('heat_equilibrium')

    def test_hw2d(self):
        demo_run('hw2d')

    def test_marker(self):
        demo_run('marker')

    def test_network_training_pytorch(self):
        demo_run('network_training_pytorch')

    def test_pipe(self):
        demo_run('pipe')

    def test_point_cloud(self):
        demo_run('point_cloud')

    def test_profile_navier_stokes(self):
        demo_run('profile_navier_stokes')

    def test_rotating_bar(self):
        demo_run('rotating_bar')

    def test_smoke_plume(self):
        demo_run('smoke_plume')
