from unittest import TestCase

from phi import struct


class MyTrait(struct.Trait):

    def __init__(self):
        struct.Trait.__init__(self, keywords=['weight'])

    def check_argument(self, struct_class, item, keyword, value):
        assert keyword == 'weight'
        assert isinstance(value, int)
        print('Checked value %s' % value)

    def endow(self, struct):
        struct._counter = 0

    def pre_validated(self, struct, item, value):
        struct._counter += item.trait_kwargs['weight']
        print(struct._counter)


@struct.definition(traits=[MyTrait()])
class Parent(struct.Struct):

    @struct.constant(default='parent')
    def parent(self, parent): return parent

    @struct.variable(default=0, dependencies='parent', weight=1)
    def density(self, density): return density


@struct.definition()
class MyStruct(Parent):

    def __init__(self, **kwargs):
        Parent.__init__(self, **struct.kwargs(locals(), include_self=False))

    @struct.constant(dependencies=['age', 'age2', 'parent'])
    def a_super_dependent(self, super_dependent): return super_dependent

    @struct.constant(default=26, dependencies=Parent.parent)
    def age(self, age): return age

    @struct.constant(dependencies='age')
    def age2(self, age2): return age2

    @struct.derived()
    def is_adult(self):
        return self.age >= 18


class TestStruct(TestCase):

    def test_custom_struct_instance(self):
        m = MyStruct()
        self.assertEqual({
            'a_super_dependent': None,
            'age': 26,
            'age2': None,
            'parent': 'parent'
        }, struct.constants(m))

    def test_batch_get(self):
        obj = {'a': [MyStruct()]}
        age = MyStruct.age(obj)
        self.assertEqual(age, {'a': [26]})
        adult = MyStruct.is_adult(obj)
        self.assertEqual(adult, {'a': [True]})
