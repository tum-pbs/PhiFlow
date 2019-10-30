import numpy as np
import os
import sys
from unittest import TestCase

import phi.viz.display as display
from phi.physics.field import Field


class PerformModelTests(display.ModelDisplay):

    def show(self):
        model = self.model
        print('Testing model %s...' % model.__class__.__name__)
        model.prepare()
        print('Model prepared.')
        self.validate_fields()
        superclasses = [b.__name__ for b in model.__class__.__bases__]
        if 'App' in superclasses:
            model.progress()
            print('Step 1 succeeded.')
            model.progress()
            print('Step 2 succeeded.')
            self.validate_fields()
        else:
            print('Skipping steps')
        model.world.reset()

    def validate_fields(self):
        for name in self.model.fieldnames:
            value = self.model.get_field(name)
            assert isinstance(value, (np.ndarray, Field)), 'Not a valid field value: %s' % value
        print('All fields are valid.')

    def play(self):
        pass


def demo_run(name):
    phiflow_path = os.path.abspath('')
    if phiflow_path.endswith('tests'):
        phiflow_path = os.path.dirname(phiflow_path)
    demos_path = os.path.join(phiflow_path, 'demos')
    if demos_path not in sys.path:
        sys.path.append(demos_path)
    display.DEFAULT_DISPLAY_CLASS = PerformModelTests
    display.AUTORUN = False
    module = __import__(name)
    try:
        module_world = getattr(module, 'world')
        module_world.reset()
    except:
        print('Could not reset world after demo %s' % name)
    print('')


class TestDemos(TestCase):

    def test_burgers_sim(self):
        demo_run('burgers_sim')

    def test_heat_equilibrium(self):
        demo_run('heat_equilibrium')

    def test_manual_smoke_numpy_or_tf(self):
        demo_run('manual_smoke_numpy_or_tf')

    def test_marker(self):
        demo_run('marker')

    def test_moving_inflow(self):
        demo_run('moving_inflow')

    def test_simpleplume(self):
        demo_run('simpleplume')

    def test_smoke_datagen_commandline(self):
        demo_run('smoke_datagen_commandline')

    def test_smoke_logo(self):
        demo_run('smoke_logo')

    def test_wavepacket(self):
        demo_run('wavepacket')
