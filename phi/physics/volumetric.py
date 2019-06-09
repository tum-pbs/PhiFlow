from .physics import *
from .domain import *


class VolumetricPhysics(Physics):

    def __init__(self, domain):
        Physics.__init__(self)
        self.domain = domain
        # Cache
        self._cached_worldstate = None
        self._cached_domainstate = None

    @property
    def grid(self):
        return self.domain.grid

    @property
    def dimensions(self):
        return self.grid.dimensions

    @property
    def rank(self):
        return self.grid.rank

    @property
    def domainstate(self):
        if self.worldstate != self._cached_worldstate:  # TODO only check inflows and obstacles
            mask = 1 - geometry_mask(self.worldstate, self.domain.grid, 'obstacle')
            self._cached_domainstate = DomainState(self.domain, self.worldstate, active=mask, accessible=mask)
            self._cached_worldstate = self.worldstate
        return self._cached_domainstate

