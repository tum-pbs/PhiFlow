import six


class Trait(object):
    """
Traits are always handled in the order in which they were declared.
Inherited traits come before newly declared structs, unless explicitly re-declared.
    """

    def __init__(self, keywords=()):
        self.keywords = keywords
        for keyword in keywords:
            assert isinstance(keyword, six.string_types)

    def check_argument(self, struct_class, item, keyword, value):
        """
Called when a keyword of the trait is used on an item.
        :param struct_class: struct class
        :param item: Item (usually created via @variable or @constant)
        :param keyword: keyword present on item and part of the trait (string)
        :param value: value associated with the keyword for the given item
        """
        pass

    def endow(self, struct):
        """
Called on newly created Structs with this trait.
This method is called before the first validation.
        :param struct: struct instance to be endowed with the trait
        """
        pass

    def pre_validate_struct(self, struct):
        """
Called before a struct instance with the trait is validated.
        :param struct: struct about to be validated
        """
        pass

    def post_validate_struct(self, struct):
        """
Called after a struct instance with the trait is validated.
        :param struct: validated struct
        """
        pass

    def pre_validated(self, struct, item, value):
        """
Processes the value of an item before the validation function is called.
        :param struct: struct undergoing validation
        :param item: item being validated
        :param value: item value before validation
        :return: processed item value which will be passed to the validation function
        """
        return value

    def post_validated(self, struct, item, value):
        """
Processes the value of an item after the validation function is called.
        :param struct: struct undergoing validation
        :param item: item being validated
        :param value: item value after validation
        :return: processed item value which will be stored in the struct
        """
        return value

    def __eq__(self, other):
        return self.__class__ == other.__class__ if isinstance(other, Trait) else False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '%s (Trait)' % self.__class__.__name__

    def __hash__(self):
        return hash(self.__class__)
