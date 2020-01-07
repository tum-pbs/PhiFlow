from __future__ import print_function

import threading
import time

from phi.physics.field.staggered_grid import StaggeredGrid

from .stream import DerivedStream


class Interleave(DerivedStream):

    def __init__(self, fields):
        DerivedStream.__init__(self, fields)
        self.field_count = len(fields)

    def shape(self, datasource):
        return self.input_fields[0].shape(datasource)

    def size(self, datasource):
        return sum([f.size(datasource) for f in self.input_fields])

    def get(self, datasource, indices):
        for index in indices:
            yield self.input_fields[index % self.field_count].get(datasource, index // self.field_count)


class Transform(DerivedStream):

    def __init__(self, transformation, field):
        DerivedStream.__init__(self, [field])
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

    def get_batch(self, streams=None, subrange=None):
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
