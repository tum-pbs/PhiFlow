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