from phi import struct


class _PROPAGATOR(object):
    ALL_OPERATIONS = 'transform'
    LINEAR_OPERATIONS = 'linear'  # scaling or +/- with other field of same flag
    RESAMPLE = 'resample'
    CHILDREN = 'children'


class _FIELD_TYPE(object):
    SCALAR = 'scalar'
    VECTOR = 'vector'
    ANY = 'any'

    @staticmethod
    def list(spatial_rank, components):
        result = [_FIELD_TYPE.ANY]
        if components == 1:
            result.append(_FIELD_TYPE.SCALAR)
        if components == spatial_rank:
            result.append(_FIELD_TYPE.VECTOR)
        return result


class Flag(struct.Struct):

    __struct__ = struct.Def([], ['_name'])

    def __init__(self, name, is_data_bound, is_structure_bound, propagators=(), field_types=()):
        self._name = name
        self._propagators = tuple(propagators)
        self._field_types = tuple(field_types)
        self._is_data_bound = is_data_bound
        self._is_structure_bound = is_structure_bound

    @property
    def name(self):
        return self._name

    @property
    def is_data_bound(self):
        return self._is_data_bound

    @property
    def is_structure_bound(self):
        return self._is_structure_bound

    def is_applicable(self, spatial_rank, component_count):
        for type in _FIELD_TYPE.list(spatial_rank, component_count):
            if type in self._field_types:
                return True
        return False

    def propagates(self, propagator):
        return propagator in self._propagators

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name


class _DivergenceFree(Flag): pass


DIVERGENCE_FREE = _DivergenceFree('divergence-free', True, False,
                                  propagators=[_PROPAGATOR.LINEAR_OPERATIONS, _PROPAGATOR.RESAMPLE],
                                  field_types=[_FIELD_TYPE.VECTOR])


class _L2Norm(Flag): pass


L2_NORMALIZED = _L2Norm('L2-normalized', True, False,
                        propagators=[],
                        field_types=[_FIELD_TYPE.ANY])
