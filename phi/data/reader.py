import math
from bisect import bisect_left

import six

try:
    # Python 3
    from collections.abc import Iterable
except ImportError:
    # Python 2.7
    from collections import Iterable
from sys import getsizeof

import numpy as np
from phi import struct

from .stream import DataStream, SourceStream

SKIP = 'skip'
WRAP = 'wrap'
CLIP = 'clip'


class BatchReader(object):

    def __init__(self, dataset, fields):
        self._dataset = dataset
        self._index = 0
        self._streams = []
        self._fields = fields
        self.streams = tuple(filter(lambda x: isinstance(x, DataStream) or isinstance(x, six.string_types), struct.flatten(fields)))
        self.stream_mask = struct.map(lambda x: x in self.streams, self._fields, content_type='stream_mask')
        for stream in self.streams:
            if isinstance(stream, DataStream):
                self._streams.append(stream)
            elif isinstance(stream, six.string_types):
                self._streams.append(SourceStream(stream))
            else:
                assert False
        self._cache = _BatchCache()
        self.indexcache = None
        self._dataset_changed()

    @property
    def dataset(self):
        return self._dataset

    def _get_batch(self, indices):
        data_list = self._cache.get(indices, self._load, add_to_cache=True)
        data = list_swap_axes(data_list)
        data_map = {self.streams[i]: data[i] for i in range(len(self._streams))}
        return struct.map(lambda x, is_stream: data_map[x] if is_stream else x, struct.zip([self._fields, self.stream_mask]), content_type=struct.INVALID)

    def _load(self, indices):
        result = []
        for index in indices:
            arrays = []
            for stream in self._streams:
                source, local_index = self.indexcache.get_source_and_local_index(index)
                data = stream.get(source, [local_index])
                array = next(iter(data))  # TODO group frames by source before calling get
                arrays.append(array)
            result.append(arrays)
        return result

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._get_batch([item])
        elif isinstance(item, slice):
            stop = len(self) if item.stop is None else item.stop
            if stop < 0:
                stop = len(self) + stop
            start = 0 if item.start is None else item.start
            step = 1 if item.step is None else item.step
            return self._get_batch(list(range(start, stop, step)))
        elif isinstance(item, Iterable):
            return self._get_batch(item)
        else:
            raise ValueError("Illegal index: %s" % item)

    def __len__(self):
        return self._len

    def _dataset_changed(self):
        self._cache.clear()
        # Compute length
        if len(self._streams) == 0:
            self._len = 0
            self.indexcache = None
        else:
            stream = self._streams[0]
            self._len = np.sum([stream.size(source, lookup=True) for source in self._dataset.sources], dtype=np.int64)
            self.indexcache = _IndexCache(self._dataset.sources, self._streams[0])

    def all_batches(self, batch_size=1, last=CLIP, loop=False):
        return _AdaptiveBatchIterator(self, batch_size, last, loop)


class _IndexCache(object):

    def __init__(self, sources, stream, randomize_scene_order=False, randomize_frame_order=False, randomize_indices=True):
        if randomize_scene_order and not randomize_indices:
            # python2 doesnt yet support indexing via array, manually permutate
            self.sources = []
            p = np.random.permutation(len(sources))
            for i in range(len(sources)):
                self.sources.append(sources[p[i]])
            # replace sometime with original code: self.sources = sources[np.random.permutation(len(sources))]
        else:
            self.sources = sources
        self.datastream = stream
        self.accumulated_sizes = []

    def get_source_and_local_index(self, index):
        max_size = 0 if len(self.accumulated_sizes) == 0 else self.accumulated_sizes[-1]
        while index >= max_size:
            source = self.sources[len(self.accumulated_sizes)]
            source_size = self.datastream.size(source, lookup=True)
            max_size = max_size + source_size
            self.accumulated_sizes.append(max_size)
        pos = bisect_left(self.accumulated_sizes, index + 1)
        local_index = index if pos == 0 else index - self.accumulated_sizes[pos - 1]
        return self.sources[pos], local_index


class _BatchCache(object):

    def __init__(self, capacity=512 * 1024 * 1024):
        self._access_order = []  # most recent ones last
        self._data_by_stream_by_index = {}
        self._size = 0
        self.capacity = capacity

    def get(self, indices, lookup_function, add_to_cache=True):
        indices = np.array(indices)
        are_cached = np.array([index in self._data_by_stream_by_index for index in indices])
        uncached_indices = indices[~are_cached]
        cached_indices = indices[are_cached]

        if len(uncached_indices) > 0:
            uncached_data = lookup_function(uncached_indices)
            if add_to_cache:
                for index, data in zip(uncached_indices, uncached_data):
                    self.add(index, data)
        else:
            uncached_data = []

        for index in cached_indices:
            self._access_order.remove(index)
            self._access_order.append(index)

        result = []
        for index in indices:
            if index in cached_indices:
                result.append(self._data_by_stream_by_index[index])
            else:
                result.append(uncached_data[int(np.where(uncached_indices == index)[0])])
        return result

    def clear(self):
        self._access_order = []
        self._data_by_stream_by_index = {}
        self._size = 0

    def add(self, index, data):
        data_size = np.sum([getsizeof(array) for array in data])
        self.remove_old(self.capacity - data_size)
        self._access_order.append(index)
        assert index not in self._data_by_stream_by_index
        self._data_by_stream_by_index[index] = data
        self._size += data_size

    def remove_old(self, target_capacity):
        if target_capacity < 0:
            target_capacity = 0
        while self._size > target_capacity:
            index = self._access_order.pop(0)
            data = self._data_by_stream_by_index.pop(index)
            for array in data:
                self._size -= getsizeof(array)


class _AdaptiveBatchIterator(object):

    def __init__(self, batchreader, batch_size, last, loop):
        assert isinstance(batchreader, BatchReader)
        assert batch_size > 0
        assert last in (SKIP, CLIP, WRAP)
        assert isinstance(loop, bool)
        self.reader = batchreader
        self.batch_size = batch_size
        self.last = last
        self.loop = loop
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        start = self.index
        stop = start + self.batch_size

        if self.last == CLIP:
            stop = min(stop, len(self.reader))
            if start == stop:
                if not self.loop:
                    raise StopIteration()
                else:
                    start = 0
                    stop = min(self.batch_size, len(self.reader))
            indices = range(start, stop)
        elif self.last == SKIP:
            if stop > len(self.reader):
                start = 0
                stop = self.batch_size
                if stop > len(self.reader):
                    if not self.loop:
                        raise StopIteration()
                    else:
                        raise AssertionError("Looping iterator with 0 batches")
            indices = range(start, stop)
        elif self.last == WRAP:
            # Repeat first frames at the end
            indices = range(start, start + self.batch_size)
            indices = [i % len(self.reader) for i in indices]
            stop = (start + self.batch_size) % len(self.reader)
        else:
            raise AssertionError()

        batch = self.reader[indices]
        self.index = stop
        return batch

    next = __next__

    def __len__(self):
        assert not self.loop, "Looping iterator has no finite length"
        if self.last == SKIP:
            return len(self.reader) // self.batch_size
        if self.last == CLIP or self.last == WRAP:
            return int(math.ceil(float(len(self.reader)) / self.batch_size))


def list_swap_axes(list_in, concatenate=True):
    if len(list_in) == 0:
        return list_in
    result = []
    for i in range(len(list_in[0])):
        subresult = []
        for sublist in list_in:
            subresult.append(sublist[i])
        if concatenate:
            subresult = np.concatenate(subresult, axis=0)
        result.append(subresult)
    return result
