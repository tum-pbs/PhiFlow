from unittest import TestCase
from phi.flow import *


def generate_test_structs():
    return [
            StaggeredGrid('Staggered Grid'),
            SmokeState('Density', 'Velocity')
        ]


class TestStruct(TestCase):
    def test_disassemble(self):
        for struct in generate_test_structs():
            tensors, reassemble = disassemble(struct)
            struct2 = reassemble(tensors)
            self.assertEqual(struct, struct2)

    def test_attributes(self):
        struct = attributes(StaggeredGrid(None))
        self.assertEqual(struct.staggered, 'staggered')

        struct = attributes(SmokeState(None, None), remove_prefix=True, qualified_names=True)
        self.assertEqual(struct.density, 'density')
        self.assertEqual(struct.velocity.staggered, 'velocity.staggered')

        struct = attributes(SmokeState(None, None), remove_prefix=True, qualified_names=False)
        self.assertEqual(struct.density, 'density')
        self.assertEqual(struct.velocity.staggered, 'staggered')