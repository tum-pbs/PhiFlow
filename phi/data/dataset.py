import os
import warnings

from .fluidformat import Scene
from .source import DataSource, SceneSource


class Dataset(object):

    def __init__(self, name, sources=()):
        self.name = name
        self.sources = list(sources)

    def __repr__(self):
        return self.name

    def add(self, datasource):
        warnings.warn("Editing datasets is deprecated.", DeprecationWarning)
        if isinstance(datasource, DataSource):
            self.sources.append(datasource)
        else:
            for source in datasource:
                self.add(source)

    def remove(self, datasource):
        warnings.warn("Editing datasets is deprecated.", DeprecationWarning)
        self.sources.remove(datasource)

    def count(self, lookup_unknown=True):
        total = 0
        for datasource in self.sources:
            s = datasource.size(lookup=lookup_unknown)
            if s is not None:
                total += s
        return total

    def __iadd__(self, other):
        warnings.warn("Editing datasets is deprecated.", DeprecationWarning)
        if isinstance(other, DataSource):
            self.add(other)
        if isinstance(other, Dataset):
            self.sources = self.sources + other.sources
        return self

    def __add__(self, other):
        newset = Dataset("%s + %s" % (self.name, other.name))
        if isinstance(other, DataSource):
            newset.sources = self.sources + [other]
        if isinstance(other, Dataset):
            newset.sources = self.sources + other.sources
        return newset

    @staticmethod
    def load(directory, indices=None, name=None, max_scenes=None, assume_same_frames=True, assume_same_shapes=True, frames=None):
        if name is None:
            name = os.path.basename(directory)
        # --- Discover scene directories ---
        index_filter = None if indices is None else lambda i: i in indices
        scenes = Scene.list(directory, max_count=max_scenes, indexfilter=index_filter)
        sources = []
        # --- Discover frame count and  array shapes ---
        shapes = dict() if assume_same_shapes else None
        for scene in scenes:
            if assume_same_frames and frames is None:
                frames = scene.frames
            sources.append(SceneSource(scene, frames=frames, shape_map=shapes))
        # --- Create Dataset ---
        if len(sources) == 0:
            raise ValueError("No data sets found in '%s' " % directory)
        return Dataset(name, sources)
