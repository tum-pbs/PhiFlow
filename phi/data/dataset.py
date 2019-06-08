from .source import *


class Dataset(object):

    def __init__(self, name):
        self.name = name
        self.sources = []

    def __repr__(self):
        return self.name

    def add(self, datasource):
        self.sources.append(datasource)

    def remove(self, datasource):
        self.sources.remove(datasource)

    def size(self, lookup=True):
        total = 0
        for datasource in self.sources:
            s = datasource.size(lookup=lookup)
            if s is not None:
                total += s
        return total

    def __iadd__(self, other):
        if isinstance(other, DataSource):
            self.add(other)
        if isinstance(other, Dataset):
            self.sources = self.sources + other.sources
        return self

    def __add__(self, other):
        newset = Dataset("%s + %s" % (self.name, other.name))
        if isinstance(other, DataSource):
            self.add(other)
        if isinstance(other, Dataset):
            self.sources = self.sources + other.sources
        return self

    @staticmethod
    def load(directory, dataset_name=None, indices=None, max_scenes=None, assume_same_frames=True, assume_same_shapes=True):
        import os
        from .fluidformat import scenes

        if dataset_name is None:
            dataset_name = os.path.basename(directory)

        dataset = Dataset(dataset_name)

        shape_map = dict() if assume_same_shapes else None
        frames = None

        indexfilter = None if indices is None else lambda i: i in indices
        scene_iterator = scenes(directory, max_count=max_scenes, indexfilter=indexfilter)

        for scene in scene_iterator:
            if assume_same_frames and frames is None:
                frames = scene.indices
            dataset.add(SceneSource(scene, frames=frames, shape_map=shape_map))

        return dataset


# def split():
#     def need_factor(self, total_size, add_count):
#         if isinstance(self.target_size, float):
#             if not self.sources:
#                 return self.target_size
#             else:
#                 return self.target_size - (self.size+add_count) / float(total_size)
#         else:
#             raise NotImplementedError()


