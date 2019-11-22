from unittest import TestCase

from phi import struct
from phi.struct.structdef import _UNUSED_ITEMS, get_type


@struct.definition()
class Parent(struct.Struct):

    @struct.prop(default='parent')
    def parent(self, parent): return parent

    @struct.attr(default=0, dependencies='parent')
    def density(self, density): return density


@struct.definition()
class MyStruct(Parent):

    def __init__(self, **kwargs):
        Parent.__init__(self, **struct.kwargs(locals(), include_self=False))
        print(self.__class__.__struct__)

    @struct.prop(dependencies=['age', 'age2', 'parent'])
    def a_super_dependent(self, super_dependent): return super_dependent

    @struct.prop(default=26, dependencies=Parent.parent)
    def age(self, age): return age

    @struct.prop(dependencies='age')
    def age2(self, age2): return age2


class TestStruct(TestCase):

    def test_custom_struct_typedef(self):
        self.assertEqual(len(_UNUSED_ITEMS), 0)
        structtype = get_type(MyStruct)
        self.assertIsNotNone(structtype)

    def test_custom_struct_instance(self):
        m = MyStruct()
        self.assertEqual({
            'a_super_dependent': None,
            'age': 26,
            'age2': None,
            'parent': 'parent'
        }, struct.properties(m))
