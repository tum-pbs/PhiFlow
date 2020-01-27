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


@struct.definition()
class Flag(struct.Struct):

    def __init__(self, name, is_data_bound, is_structure_bound, propagators=(), field_types=(), **kwargs):
        struct.Struct.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def name(self, name): return name

    @struct.constant()
    def is_data_bound(self, v):
        assert isinstance(v, bool)
        return v

    @struct.constant()
    def is_structure_bound(self, v):
        assert isinstance(v, bool)
        return v

    @struct.constant()
    def propagators(self, propagators):
        return tuple(propagators)

    @struct.constant()
    def field_types(self, field_types):
        return tuple(field_types)

    def is_applicable(self, spatial_rank, component_count):
        for type in _FIELD_TYPE.list(spatial_rank, component_count):
            if type in self.field_types:
                return True
        return False

    def propagates(self, propagator):
        return propagator in self.propagators

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name


@struct.definition()
class _DivergenceFree(Flag):
    pass


DIVERGENCE_FREE = _DivergenceFree('divergence-free', True, False,
                                  propagators=[_PROPAGATOR.LINEAR_OPERATIONS, _PROPAGATOR.RESAMPLE],
                                  field_types=[_FIELD_TYPE.VECTOR])


@struct.definition()
class _L2Norm(Flag):
    pass


L2_NORMALIZED = _L2Norm('L2-normalized', True, False,
                        propagators=[],
                        field_types=[_FIELD_TYPE.ANY])


@struct.definition()
class SamplePoints(Flag):
    pass


SAMPLE_POINTS = SamplePoints('sample-points', True, True,
                             propagators=[],
                             field_types=[_FIELD_TYPE.VECTOR])
