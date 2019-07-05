from unittest import TestCase
from phi.flow import *
from phi.math.struct import *


def generate_test_structs():
    return [
            StaggeredGrid('Staggered Grid'),
            [StaggeredGrid('Staggered Grid')],
            [('Item',)],
        {'A': 'Entry A', 'Vel': StaggeredGrid('v')},
        CollectiveState((Smoke(density='density', velocity='velocity'),))
        ]


class TestStruct(TestCase):
    def test_identity(self):
        for struct in generate_test_structs():
            struct2 = map(lambda s: s, struct, recursive=False)
            self.assertEqual(struct, struct2)
            struct3 = map(lambda t: t, struct, recursive=True)
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
            names = flatten(map(lambda attr: attr.name, struct, trace=True))
            self.assertGreater(len(names), 0)
            for name in names:
                self.assertIsInstance(name, str)

    def test_paths(self):
        struct = {'Vels': [StaggeredGrid('v')]}
        names = flatten(map(lambda attr: attr.path(), struct, trace=True))
        self.assertEqual(names[0], 'Vels.0.staggered')

    def test_copy(self):
        smoke = Smoke(density='Density', velocity='Velocity')
        v = smoke.copied_with(velocity=StaggeredGrid('V2'))
        self.assertEqual(v.velocity.staggered, 'V2')
        self.assertEqual(v.density, 'Density')

        d = smoke.copied_with(density='D2')
        self.assertEqual(d.density, 'D2')

    def test_zip(self):
        a = StaggeredGrid('a')
        b = StaggeredGrid('b')
        stacked = map(lambda *x: x, zip([a, b]))
        numpy.testing.assert_equal(stacked.staggered, ('a', 'b'))