from ._context import _struct_context, _STRUCT_CONTEXT_STACK
from ._structdef import Item


class ItemCondition(object):
    """
    ItemConditions are used to filter struct items.
    They represent a named boolean function on items.
    
    In addition, they can be used in 'with ItemCondition:' blocks, adding the condition to the thread context for all actions within that block.
    In particular, struct.map, Struct.shape, Struct.staticshape are affected by context conditions.
    
    This module provides some standard conditions like ALL_ITEMS, DATA, VARIABLES, CONSTANTS.

    Args:

    Returns:

    """

    def __init__(self, item_condition, name=None):
        assert item_condition is None or callable(item_condition), item_condition
        self.item_condition = item_condition
        if name is not None:
            self.name = name
        else:
            self.name = item_condition.__name__ if item_condition is not None else 'ALL'

    def condition_check(self, item):
        return True if self.item_condition is None else self.item_condition(item)

    __call__ = condition_check

    def __enter__(self):
        self.context = _struct_context(self)
        return self.context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self.context.__exit__(exc_type, exc_val, exc_tb)
        self.context = None
        return result

    def __repr__(self):
        return self.name

    def __and__(self, other):
        assert isinstance(other, ItemCondition)
        return ItemCondition(lambda item: self(item) and other(item), name='%s and %s' % (self.name, other.name))

    def __or__(self, other):
        assert isinstance(other, ItemCondition)
        return ItemCondition(lambda item: self(item) or other(item), name='%s or %s' % (self.name, other.name))


CONSTANTS = ItemCondition(lambda item: not item.is_variable, 'CONSTANTS')
VARIABLES = ItemCondition(lambda item: item.is_variable, 'VARIABLES')
DATA = ItemCondition(lambda item: item.holds_data, 'DATA')
ALL_ITEMS = ItemCondition(None)


def context_item_condition(item):
    """
    Checks all thread-global item conditions.
    Conditions can be specified using 'with ItemCondition:' blocks.
    If no condition was specified, this function defaults to testing whether the item holds data.

    Args:
      item: item to be checked

    Returns:
      True if the item passes all conditions, False otherwise

    """
    user_specified = False
    for context in _STRUCT_CONTEXT_STACK:
        if isinstance(context, ItemCondition):
            user_specified = True
            if not context.condition_check(item):
                return False
    if user_specified:
        return True
    else:
        return item.holds_data  # default condition if none specified


def ignore(items):
    if not isinstance(items, (tuple, list)):
        items = (items,)
    for ignored_item in items:
        assert isinstance(ignored_item, Item) or isinstance(ignored_item, str)

    def is_ignored(item):
        for ignored in items:
            if item is ignored:
                return True
            if item.name == ignored:
                return True
        return False

    condition = ItemCondition(lambda item: not is_ignored(item), 'ignore %s' % items)
    return condition
