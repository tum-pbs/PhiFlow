

class Trait(object):
    """
    Traits are always handled in the order in which they were declared.
    Inherited traits come before newly declared structs, unless explicitly re-declared.

    Args:

    Returns:

    """

    def __init__(self, keywords=()):
        self.keywords = keywords
        for keyword in keywords:
            assert isinstance(keyword, str)

    def check_argument(self, struct_class, item, keyword, value):
        """
        Called when a keyword of the trait is used on an item.

        Args:
          struct_class: struct class
          item: Item (usually created via @variable or @constant)
          keyword: keyword present on item and part of the trait (string)
          value: value associated with the keyword for the given item

        Returns:

        """
        pass

    def endow(self, struct):
        """
        Called on newly created Structs with this trait.
        This method is called before the first validation.

        Args:
          struct: struct instance to be endowed with the trait

        Returns:

        """
        pass

    def pre_validate_struct(self, struct):
        """
        Called before a struct instance with the trait is validated.

        Args:
          struct: struct about to be validated

        Returns:

        """
        pass

    def post_validate_struct(self, struct):
        """
        Called after a struct instance with the trait is validated.

        Args:
          struct: validated struct

        Returns:

        """
        pass

    def pre_validated(self, struct, item, value):
        """
        Processes the value of an item before the validation function is called.

        Args:
          struct: struct undergoing validation
          item: item being validated
          value: item value before validation

        Returns:
          processed item value which will be passed to the validation function

        """
        return value

    def post_validated(self, struct, item, value):
        """
        Processes the value of an item after the validation function is called.

        Args:
          struct: struct undergoing validation
          item: item being validated
          value: item value after validation

        Returns:
          processed item value which will be stored in the struct

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
