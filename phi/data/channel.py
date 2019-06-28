import six
import numpy as np
from .source import DataSource


class DataChannel(object):

    def shape(self, datasource):
        raise NotImplementedError(self)

    def size(self, datasource, lookup=False):
        raise NotImplementedError(self)

    def get(self, datasource, indices):
        raise NotImplementedError(self)

    def frames(self, datasource):
        raise NotImplementedError(self)


class SourceChannel(DataChannel):

    def __init__(self, name):
        assert isinstance(name, six.string_types)
        self._name = name

    def shape(self, datasource):
        return datasource.shape(self._name)

    def size(self, datasource, lookup=False):
        return datasource.size(lookup=lookup)

    def get(self, datasource, indices):
        frames = datasource.frames()
        frames = [frames[i] for i in indices]
        return datasource.get(self._name, frames)

    def frames(self, datasource):
        return datasource.frames()

    def __repr__(self):
        return self._name


class _SourceFrame(DataChannel):

    def shape(self, datasource):
        return (1, )

    def size(self, datasource, lookup=False):
        return datasource.size(lookup=lookup)

    def get(self, datasource, indices):
        frames = datasource.frames()
        return np.expand_dims([frames[i] for i in indices], -1)

    def frames(self, datasource):
        return datasource.frames()


FRAME = _SourceFrame()


class _SourceScene(DataChannel):

    def shape(self, datasource):
        return (1, )

    def size(self, datasource, lookup=False):
        return datasource.size(lookup=lookup)

    def get(self, datasource, indices):
        return np.expand_dims([datasource.scene] * len(indices), -1)

    def frames(self, datasource):
        return datasource.frames()


SCENE = _SourceScene()


class _Source(DataChannel):

    def shape(self, datasource):
        return (1, )

    def size(self, datasource, lookup=False):
        return datasource.size(lookup=lookup)

    def get(self, datasource, indices):
        return np.expand_dims([datasource] * len(indices), -1)

    def frames(self, datasource):
        return datasource.frames()


SOURCE = _Source()



class DerivedChannel(DataChannel):

    def __init__(self, input_channels):
        self.input_fields = [c if isinstance(c, DataChannel) else SourceChannel(c) for c in input_channels]

    def __repr__(self):
        return "%s(%s)" % (type(self), self.input_fields)


class FrameSelect(DerivedChannel):

    def __init__(self, selector, channel):
        """
Selects specific frames from the input.
        :param selector: Either a frame index, list of frame frames or a selection function mapping a list of all frames to a list of selected frames
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