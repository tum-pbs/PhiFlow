from unittest import TestCase
from phi.flow import *


def generate_test_structs():
    return [
            StaggeredGrid('Staggered Grid'),
            Smoke(density='Density', velocity='Velocity'),
            [StaggeredGrid('Staggered Grid')],
            [('Item',)],
        {'A': 'Entry A', 'Vel': StaggeredGrid('v')}
        ]


class TestStruct(TestCase):
    def test_identity(self):
        for struct in generate_test_structs():
            struct2 = Struct.map(lambda s: s, struct)
            self.assertEqual(struct, struct2)
            struct3 = Struct.flatmap(lambda t: t, struct)
            self.assertEqual(struct, struct3)

    def test_flatten(self):
        for struct in generate_test_structs():
            flat, _ = Struct.flatten(struct)
            self.assertIsInstance(flat, list)
            for item in flat:
                self.assert_(not Struct.isstruct(item), 'The result of flatten(%s) is not flat.' % struct)

    def test_names(self):
        n = Struct.mapnames(StaggeredGrid(None))
        self.assertEqual(n.staggered, 'staggered')

        n = Struct.mapnames(Smoke(density=None, velocity=None))
        self.assertEqual(n.density, 'density')
        self.assertEqual(n.velocity.staggered, 'velocity.staggered')

        n = Struct.mapnames([(None,)])
        self.assertEqual(n[0][0], '0.0')

    def test_copy(self):
        smoke = Smoke(density='Density', velocity='Velocity')
        v = smoke.copied_with(velocity=StaggeredGrid('V2'))
        self.assertEqual(v.velocity.staggered, 'V2')
        self.assertEqual(v.density, 'Density')

        d = smoke.copied_with(density='D2')
        self.assertEqual(d.density, 'D2')
