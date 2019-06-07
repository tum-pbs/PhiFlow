from .source import *
import numpy as np


class DataChannel(object):

    def __init__(self, name=None):
        self._name = name

    @property
    def name(self):
        return self._name

    def shape(self, datasource):
        raise NotImplementedError(self)

    def size(self, datasource, lookup=False):
        raise NotImplementedError(self)

    def get(self, datasource, indices):
        raise NotImplementedError(self)

    def frames(self, datasource):
        raise NotImplementedError(self)

    def __repr__(self):
        return self._name


class SourceChannel(DataChannel):

    def __init__(self, name):
        DataChannel.__init__(self, name)

    def shape(self, datasource):
        return datasource.shape(self.name)

    def size(self, datasource, lookup=False):
        return datasource.size(lookup=lookup)

    def get(self, datasource, indices):
        frames = datasource.frames()
        frames = [frames[i] for i in indices]
        return datasource.get(frames)

    def frames(self, datasource):
        return datasource.frames()


# class SourceFrame(DataChannel):
#
#     def shape(self, datasource):
#         return (1, )
#
#     def size(self, datasource, lookup=False):
#         return datasource.size(lookup=lookup)
#
#     def get(self, datasource, indices):
#         frames = datasource.frames()
#         return [frames[i] for i in indices]
#
#
# class SourceScene(DataChannel):
#
#     def shape(self, datasource):
#         return (1, )
#
#     def size(self, datasource, lookup=False):
#         return datasource.size(lookup=lookup)
#
#     def get(self, datasource, indices):
#         return [datasource.scene] * len(indices)



class DerivedChannel(DataChannel):

    def __init__(self, input_channels):
        DataChannel.__init__(self)
        self.input_fields = [c if isinstance(c, DataChannel) else SourceChannel(c) for c in input_channels]

    def __repr__(self):
        return "%s:%s(%s)" % (self.name, type(self), self.input_fields)


class FrameSelect(DerivedChannel):

    def __init__(self, selector, channel):
        """
Selects specific frames from the input.
        :param selector: Either a frame index, list of frame indices or a selection function mapping a list of all frames to a list of selected frames
        :param channel: input channel
        """
        DerivedChannel.__init__(self, [channel])
        self.channel = self.input_fields[0]
        if callable(selector):
            self.selection_function = selector
        else:
            if isinstance(selector, int):
                self.selection_function = lambda frames: [selector]
            else:
                self.selection_function = lambda frames: selector

    def shape(self, datasource):
        return self.channel.shape(datasource)

    def get(self, datasource, indices):
        input_frames = self.channel.frames(datasource)
        frames = self.selection_function(input_frames)
        if isinstance(frames, int):
            frames = [frames]
        try:
            selected_frames = [frames[i] for i in indices]
        except:
            raise ValueError("BatchSelect: selection function must return a list of integers, but got %s for frames %s" % (frames, datasource.frames()))
        return self.channel.get(datasource, selected_frames)

    def size(self, datasource, lookup=False):
        if not lookup:
            size = datasource.size()
            if size is None:
                return None
        input_frames = self.channel.frames(datasource)
        frames = self.selection_function(input_frames)
        return 1 if isinstance(frames, int) else len(frames)

    def frames(self, datasource):
        input_frames = self.channel.frames(datasource)
        frames = self.selection_function(input_frames)
        return [frames] if isinstance(frames, int) else frames