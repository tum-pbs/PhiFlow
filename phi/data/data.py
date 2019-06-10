from __future__ import print_function
from phi.data.fluidformat import *
from phi.math.nd import StaggeredGrid
import six, logging, math, itertools, threading, time




class Interleave(DerivedChannel):

    def __init__(self, fields):
        DerivedChannel.__init__(self, fields)
        self.field_count = len(fields)

    def shape(self, datasource):
        return self.input_fields[0].shape(datasource)

    def size(self, datasource):
        return sum([f.size(datasource) for f in self.input_fields])

    def get(self, datasource, indices):
        for index in indices:
            yield self.input_fields[index % self.field_count].get(datasource, index // self.field_count)


class Transform(DerivedChannel):

    def __init__(self, transformation, field):
        DerivedChannel.__init__(self, [field])
        self.field = self.input_fields[0]
        self.transformation = transformation

    def shape(self, datasource):
        return self.field.shape(datasource)

    def size(self, datasource):
        return self.field.size(datasource)

    def get(self, datasource, indices):
        for index in indices:
            yield self.transformation(self.field.get(datasource, index))



class AsyncBatchIterator(BatchIterator):

    def __init__(self, batch_iterator, logf=None):
        # type: (BatchIterator, function) -> AsyncBatchIterator
        BatchIterator.__init__(self, batch_iterator.database, batch_iterator.dataset, batch_iterator.iterator_generator, cache=False)
        self.it = batch_iterator
        self.batches = []
        self.batch_ready = threading.Event()
        self.progressed = threading.Event()
        self.alive = True
        self.logf = logf
        threading.Thread(target=self.next_loop).start()

    def build(self, fieldnames_or_placeholders):
        self.it.build(fieldnames_or_placeholders)

    def get_batch_size(self):
        return self.it.get_batch_size()

    def __len__(self):
        return len(self.it)

    @property
    def batch_count(self):
        return self.it.batch_count

    def get_batch(self, channels=None, subrange=None):
        self.batch_ready.wait()
        # print("Retrieving batch %d"  % self.index)
        if not self.batches:
            raise RuntimeError("get_batch failed in AsyncBatchIterator")
        return self.batches[0]

    def progress(self):
        BatchIterator.progress(self)
        # print("Progress to batch %d" % self.index)
        del self.batches[0]
        if not self.batches:
            self.batch_ready.clear()
        self.progressed.set()

    def __getitem__(self, item):
        raise NotImplementedError()

    def next_loop(self):
        while self.alive:
            while len(self.batches) < 2:
                time_before = time.time()
                self.batches.append(self.it.get_batch())
                duration = time.time() - time_before
                self.logf and self.logf("Retrieving batch %d took %f seconds." % (self.it.index, duration))
                self.batch_ready.set()
                self.it.progress()
            self.progressed.wait()
            self.progressed.clear()

    def __del__(self):
        self.alive = False


class LinearIterator(IndexIterator):

    def __init__(self, total, batch_size, shuffled, mode):
        assert total != 0, "Cannot create LinearIterator on empty set"
        self.total = total
        self.batch_size = batch_size
        self.shuffled = shuffled
        self.mode = mode
        if self.shuffled:
            self.perm = np.random.permutation(self.total)

    def get_batch_count(self):
        if not np.isfinite(self.total): return self.total
        if self.mode == "skip":
            return self.total // self.batch_size
        if self.mode == "pad" or self.mode == "dynamic":
            return int(math.ceil(float(self.total) / self.batch_size))
        raise ValueError("mode %s"%self.mode)


    def get_indices(self, batch_index):
        index = batch_index * self.batch_size
        if self.mode == "dynamic":
            indices = range(index, min(index + self.batch_size, self.total))
        elif self.mode == "skip":
            # Given range must be valid
            indices = range(index, index + self.batch_size)
        elif self.mode == "pad":
            # Repeat first indices at the end
            indices = range(index, index + self.batch_size)
            indices = [i % self.total for i in indices]
        else: raise ValueError("mode %s"%self.mode)
        if self.shuffled:
            indices = [self.perm[i] for i in indices]

        return indices
