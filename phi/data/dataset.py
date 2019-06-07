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
    def load(directory, dataset_name=None, max_scenes=None, assume_same_frames=True, assume_same_shapes=True):
        import os
        from .fluidformat import scenes

        if dataset_name is None:
            dataset_name = os.path.basename(directory)

        dataset = Dataset(dataset_name)

        frames = None
        shape

        scene_iterator = scenes(directory, max_count=max_scenes)
        for scene in scene_iterator:
            dataset.add(SceneSource(scene))
        pass  # TODO


# def split():
#     def need_factor(self, total_size, add_count):
#         if isinstance(self.target_size, float):
#             if not self.sources:
#                 return self.target_size
#             else:
#                 return self.target_size - (self.size+add_count) / float(total_size)
#         else:
#             raise NotImplementedError()


