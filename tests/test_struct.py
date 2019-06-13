from unittest import TestCase
from phi.flow import *


def generate_test_structs():
    return [
            StaggeredGrid('Staggered Grid'),
            SmokeState('Density', 'Velocity'),
            [StaggeredGrid('Staggered Grid')],
            [('Item',)]
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
        self.assertEqual(n.staggered, '_staggered')

        n = Struct.mapnames(SmokeState(None, None))
        self.assertEqual(n.density, '_density')
        self.assertEqual(n.velocity.staggered, '_velocity._staggered')

        n = Struct.mapnames([(None,)])
        self.assertEqual(n[0][0], 'index0.index0')