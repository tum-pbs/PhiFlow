

def first(leaves):
    return leaves[0]


def data(leaves):
    return math.stack(leaves, 0)


def object(leaves):
    return tuple(leaves)


def assert_equal(leaves):
    for leaf in leaves[1:]:
        assert leaf == leaves[0]
    return leaves[0]


def invalidate(leaves):
    return None


def revalidate(leaves):
    return validate(None)


class Batch(tuple):

    def __new__(cls, items):
        return tuple.__new__(Batch, items)
