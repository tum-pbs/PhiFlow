from phi import math
from ._geom import Geometry


def assert_same_rank(rank1, rank2, error_message):
    rank1_, rank2_ = _rank(rank1), _rank(rank2)
    if rank1_ is not None and rank2_ is not None:
        assert rank1_ == rank2_, 'Ranks do not match: %s and %s. %s' % (rank1_, rank2_, error_message)


def _rank(rank):
    if rank is None:
        return None
    elif isinstance(rank, int):
        pass
    elif isinstance(rank, Geometry):
        rank = rank.rank
    else:
        rank = math.spatial_rank(rank)
    return None if rank == 0 else rank
