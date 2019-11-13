from unittest import TestCase

from phi.math import *
from phi.flow import *
from phi.struct import *


def generate_test_structs():
        return [manta.centered_grid(np.zeros([1,4,1])),
                [('Item',)],
                {'A': 'Entry A', 'Vel': manta.staggered_grid(np.zeros([1,5,5,2]))},
                CollectiveState((Smoke(Domain([4])),))]


class TestStruct(TestCase):
    
    def test_identity(self):
        for struct in generate_test_structs():
            with anytype():
                struct2 = map(lambda s: s, struct, recursive=False)
                self.assertEqual(struct, struct2)
                struct3 = map(lambda t: t, struct, recursive=True)
                a = struct == struct3
                self.assertEqual(struct, struct3)
                struct4 = map(lambda t: t, struct, include_properties=True)
                self.assertEqual(struct, struct4)

    def test_flatten(self):
        for struct in generate_test_structs():
            flat = flatten(struct)
            self.assertIsInstance(flat, list)
            self.assertGreater(len(flat), 0)
            for item in flat:
                self.assert_(not isstruct(item), 'The result of flatten(%s) is not flat.' % struct)

    def test_names(self):
        for struct in generate_test_structs():
            with anytype():
                names = flatten(map(lambda attr: attr.name, struct, trace=True))
                self.assertGreater(len(names), 0)
                for name in names:
                    self.assertIsInstance(name, str)

    def test_paths(self):
        struct = {'Vels': [CenteredGrid('v', box[0:1], np.zeros([1, 4, 1]))]}
        with anytype():
            names = flatten(map(lambda attr: attr.path(), struct, trace=True))
        self.assertEqual(names[0], 'Vels.0.data')

    def test_copy(self):
        with anytype():
            smoke = Smoke(Domain([4]), density='Density', velocity='Velocity')
            v = smoke.copied_with(velocity='V2')
            self.assertEqual(v.velocity, 'V2')
            self.assertEqual(v.density, 'Density')

            try:
                d = smoke.copied_with(velocity='D2')
                self.fail()
            except:
                pass

    def test_zip(self):
        with anytype():
            a = CenteredGrid('a', None, 'a')
            b = CenteredGrid('b', None, 'b')
            stacked = map(lambda *x: x, zip([a, b]))
            np.testing.assert_equal(stacked.data, ('a', 'b'))
