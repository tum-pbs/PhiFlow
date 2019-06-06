from __future__ import print_function
from phi.data.fluidformat import *
from phi.math.nd import StaggeredGrid
import six, logging, math, itertools, threading, time


DATAFLAG_TRAIN = "train"
DATAFLAG_TEST = "test"


# Channels

class DataChannel(object):

    def __init__(self):
        pass

    def shape(self, datasource):
        raise NotImplementedError()

    def size(self, datasource):
        raise NotImplementedError()

    def get(self, datasource, indices):
        raise NotImplementedError()


class SourceChannel(DataChannel):

    def __init__(self, name):
        DataChannel.__init__(self)
        self.name = name

    def shape(self, datasource):
        shape = self.any_array(datasource).shape
        return tuple(shape)

    def size(self, datasource):
        return datasource.size

    def any_array(self, datasource):
        return datasource.get_any(self.name)

    def get(self, datasource, indices):
        for index in indices:
            yield datasource.get(self.name, index)

    def __str__(self):
        return self.name


class DerivedChannel(DataChannel):

    def __init__(self, input_fields):
        DataChannel.__init__(self)
        self.input_fields = [SourceChannel(f) if isinstance(f, six.string_types) else f for f in input_fields]

    def __str__(self):
        return "%s(%s)" % (type(self), self.input_fields)


class BatchSelect(DerivedChannel):

    def __init__(self, selection_function, field):
        DerivedChannel.__init__(self, [field])
        self.field = self.input_fields[0]
        if callable(selection_function):
            self.selection_function = selection_function
        else:
            if isinstance(selection_function, int):
                selection_function = [selection_function]
            self.selection_function = lambda len: selection_function

    def shape(self, datasource):
        input_shape = self.field.shape(datasource)
        return [len(self.selection_function(input_shape[0]))] + list(input_shape)[1:]

    def get(self, datasource, indices):
        selected_batches = self.selection_function(self.field.size(datasource))
        if isinstance(selected_batches, int):
            selected_batches = [selected_batches]
        try:
            prop_indices = [selected_batches[i] for i in indices]
        except:
            raise ValueError("BatchSelect: selection function must return a list of integers, but got %s for indices %s" % (selected_batches, indices))
        return self.field.get(datasource, prop_indices)

    def size(self, datasource):
        return len(self.selection_function(self.field.size(datasource)))


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


# Lookup tables

class IndexLookupTable(object):

    def lookup(self, field, indices):
        raise NotImplementedError()

    def get_index_count(self):
        raise NotImplementedError()


class DirectIndexLookupTable(IndexLookupTable):

    def __init__(self, fields, dataset):
        self.dataset = dataset
        lengths = [fields[0].size(source) for source in dataset.sources]
        self.total = sum(lengths)
        if lengths:
            self.accum = [lengths[0]]
            for l in lengths[1:]:
                self.accum.append(self.accum[-1] + l)
        else:
            self.accum = []

    def lookup(self, field, indices):
        logging.debug("Looking up field %s at %s" % (field, indices))
        idx_by_source = self.decode_indices(indices)
        generators = []
        for datasource, indices in idx_by_source.items():
            generators.append(field.get(datasource, indices))
        return itertools.chain(*generators)

    def decode_indices(self, indices):
        idx_by_source = {}
        for index in indices:
            datasource, dec_index = self.decode_index(index)
            if datasource in idx_by_source:
                idx_by_source[datasource].append(dec_index)
            else:
                idx_by_source[datasource] = [dec_index]
        return idx_by_source

    def decode_index(self, index):
        source_index = 0
        while index >= self.accum[source_index]:
            source_index += 1
        datasource = self.dataset.sources[source_index]
        if source_index > 0:
            index = index - self.accum[source_index - 1]
        return datasource, index

    def get_index_count(self):
        return self.total


class CachedIndexLookupTable(IndexLookupTable):

    def __init__(self, fields, dataset):
        self.lookup_table = DirectIndexLookupTable(fields, dataset)
        self.cache = {f: [None for i in range(self.lookup_table.total)] for f in fields}

    def lookup(self, field, indices):
        cache = self.cache[field]
        cached_arrays = [cache[index] for index in indices]
        if None_in(cached_arrays):
            missing_indices = [indices[i] for i in range(len(indices)) if cached_arrays[i] is None]
            missing_arrays = self.lookup_table.lookup(field, missing_indices)
            for missing_index, missing_array in zip(missing_indices, missing_arrays):
                cache[missing_index] = missing_array
            return [cache[index] for index in indices]
        else:
            return cached_arrays

    def get_index_count(self):
        return self.lookup_table.get_index_count()


def None_in(list):
    for element in list:
        if element is None:
            return True
    return False


# Core classes

class DataSource(object):

    def __init__(self, flags):
        self.flags = flags

    def get(self, fieldname, index):
        raise NotImplementedError()

    def get_any(self, fieldname):
        raise NotImplementedError()


class SceneSource(DataSource):

    def __init__(self, scene, indices, flags):
        DataSource.__init__(self, flags)
        self.scene = scene
        self.indices = indices

    @property
    def size(self):
        return len(self.indices)

    def get(self, fieldname, index):
        real_index = self.indices[index]
        logging.debug("Reading field %s at %d from scene %s" % (fieldname, real_index, self.scene))
        data = self.scene.read_array(fieldname, real_index)
        return data

    def get_any(self, fieldname):
        return self.get(fieldname, self.indices[0])


class GeneratorSource(DataSource):

    def __init__(self, size, fieldname_to_generator_dict, flags):
        DataSource.__init__(self, flags)
        self.size = size
        self.generators = fieldname_to_generator_dict

    def get(self, fieldname, index):
        return self.generators[fieldname](index)

    def get_any(self, fieldname):
        return self.generators[fieldname](0)


class Dataset(object):

    def __init__(self, name, target_size, flags):
        self.name = name
        self.target_size = target_size
        self.sources = []
        self.flags = flags

    @property
    def size(self):
        return sum([source.size for source in self.sources])

    def __repr__(self):
        return self.name

    def need_factor(self, total_size, add_count):
        if isinstance(self.target_size, float):
            if not self.sources:
                return self.target_size
            else:
                return self.target_size - (self.size+add_count) / float(total_size)
        else:
            raise NotImplementedError()


class Database(object):

    def __init__(self, *datasets):
        if datasets:
            self.sets = { set.name: set for set in datasets}
        else:
            self.sets = {"default": Dataset("default", 1.0, (DATAFLAG_TRAIN, DATAFLAG_TEST))}
        self.fields = {}  # name to DataChannel map
        self.names = {}  # DataChannel to name map
        self.scene_count = 0

    def add(self, name, field=None):
        if isinstance(name, (tuple, list)):
            if field is None:
                for n in name:
                    self.add(n)
            else:
                for n, f in zip(name, field):
                    self.add(n, f)

        else:
            assert name not in self.fields, "Name %s already exists." % name
            if field is None:
                field = SourceChannel(name)
            self.fields[name] = field
            self.names[field] = name

    def field_lookup(self, fieldname):
        return self.fields[fieldname]

    def name_lookup(self, field):
        return self.names[field]

    def put_scenes(self, scenes, per_scene_indices=None, dataset=None, allow_scene_split=False, logf=None):
        count = 0
        for scene in scenes:
            count += 1
            if dataset is not None:
                self.put_scene(scene, per_scene_indices, dataset, logf)
            else:
                if allow_scene_split:
                    self.put_scene(scene, per_scene_indices, dataset=None, logf=logf)
                else:
                    indices = per_scene_indices if per_scene_indices else scene.indices
                    datasets = list(self.sets.values())
                    needs = [d.need_factor(self.total_size(), len(indices)) for d in datasets]
                    set_max_need = datasets[ needs.index(max(needs)) ]
                    self.put_scene(scene, indices, dataset=set_max_need, logf=logf)
        logf and logf("Added %d scenes to database." % count)

    def put_scene(self, scene, indices=None, dataset=None, logf=None):
        if not indices:
            indices = scene.indices
            logf and logf("Adding %d frames from scene %s" % (len(indices), scene))
        self.scene_count += 1

        if dataset:
            if not isinstance(dataset, Dataset):
                dataset = self.sets[dataset]
            dataset.sources.append(SceneSource(scene, indices, dataset.flags))
        else:
            datasets = list(self.sets.values())
            needs = np.array([d.need_factor(self.total_size(), len(indices)) for d in datasets])
            needs /= sum(needs)
            i = 0
            for dataset, need in zip(datasets, needs):
                n = int(round(need * len(indices)))
                index_range = range(i, min(len(indices), i+n))
                dataset.sources.append(SceneSource(scene, index_range, dataset.flags))
                logf and logf("Adding %d frames from scene %s to set %s" % (len(index_range), scene, dataset))
                i += n
                if i >= len(indices):
                    break

    def put_generated(self, fieldname_to_generator_dict, dataset=None):
        if dataset:
            gen = GeneratorSource(np.inf, fieldname_to_generator_dict, dataset.flags)
            dataset.sources.append(gen)
        else:
            for dataset in self.sets.values():
                self.put_generated(fieldname_to_generator_dict, dataset)

    def total_size(self):
        return sum([set.size for set in self.sets.values()])

    def linear_iterator(self, dataset, fieldnames, batch_size, shuffled=False, mode="dynamic", async_load=True, logf=None):
        iterator = BatchIterator(self, self.sets[dataset], lambda lookup: LinearIterator(lookup.get_index_count(), batch_size, shuffled, mode))
        return self._decorate_iterator(iterator, fieldnames, async_load, logf)

    def fixed_range(self, dataset, fieldnames, selection_indices=None):
        iterator = BatchIterator(self, self.sets[dataset], lambda lookup: FixedRange(lookup.get_index_count(), selection_indices))
        return self._decorate_iterator(iterator, fieldnames, False, None)

    def _decorate_iterator(self, iterator, fieldnames, async_load, logf):
        iterator.build(fieldnames)
        if async_load:
            iterator = AsyncBatchIterator(iterator, logf=logf)
        return iterator


def contained_batch_dimension(dict):
    any_array = six.next(six.itervalues(dict))
    return any_array.shape[0]


class BatchIterator(object):

    def __init__(self, database, dataset, iterator_generator, cache=True):
        self.database = database
        self.dataset = dataset
        self.lookup_table = None
        self.iterator_generator = iterator_generator
        self.iterator = None
        self.index = 0
        self.cache = cache and np.isfinite(self.dataset.size)
        if self.cache != cache:
            logging.warn("Caching was disabled because dataset has an invalid size")

    def build(self, fieldnames_or_placeholders):
        fieldnames = placeholders_to_fieldnames(fieldnames_or_placeholders)
        self.fields = [self.database.field_lookup(f) for f in fieldnames]
        if self.cache:
            self.lookup_table = CachedIndexLookupTable(self.fields, self.dataset)
        else:
            self.lookup_table = DirectIndexLookupTable(self.fields, self.dataset)
        self.iterator = self.iterator_generator(self.lookup_table)

    def progress(self):
        self.index += 1
        total = self.batch_count
        if total > 0 and np.isfinite(total):
            self.index = self.index % total

    def get_batch(self, channels=None, subrange=None):
        indices = self.iterator.get_indices(self.index)
        assert len(indices) > 0, "Iterator %s returned no indices for index %d" % (self.iterator, self.index)
        if subrange:
            indices = [indices[i] for i in subrange]
        field_to_data_dict = self.multi_lookup(indices)
        return {self.database.name_lookup(f): data for (f, data) in field_to_data_dict.items()}

    def fill_feed_dict(self, feed_dict, placeholders, subrange=None):
        batch = self.get_batch(placeholders_to_fieldnames(placeholders), subrange=subrange)
        self.progress()

        for placeholder in placeholders:
            if isinstance(placeholder, StaggeredGrid):
                placeholder = placeholder.staggered
            if not placeholder.name.endswith(":0"):
                raise ValueError("Not a valid placeholder: %s"%placeholder)
            name = placeholder.name[:-2]
            array = batch[name]
            feed_dict[placeholder] = array
        return feed_dict

    def get_batch_size(self):
        return len(self.iterator.get_indices(self.index))

    def __len__(self):
        return self.iterator.get_batch_count()

    @property
    def batch_count(self):
        return self.iterator.get_batch_count()

    def __getitem__(self, item):
        indices = self.iterator.get_indices(self.index)
        return self.multi_lookup(indices)

    def multi_lookup(self, indices):
        return {field: np.concatenate(list(self.lookup_table.lookup(field, indices))) for field in self.fields}


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




def placeholders_to_fieldnames(placeholders):
    for placeholder in placeholders:
        if isinstance(placeholder, six.string_types):
            yield placeholder
        else:
            if isinstance(placeholder, StaggeredGrid):
                placeholder = placeholder.staggered
            try:
                if placeholder.name.endswith(":0"):
                    yield placeholder.name[:-2]
            except:
                raise ValueError("Not a string or placeholder: %s of type %s" % (placeholder, type(placeholder)))


# Index Iterators

class IndexIterator(object):

    def get_indices(self, batch_index):
        raise NotImplementedError()

    def get_batch_count(self):
        raise NotImplementedError()


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
        else: raise ValueError("mode %s"%self.mode
                               )
        if self.shuffled:
            indices = [self.perm[i] for i in indices]

        return indices


class FixedRange(IndexIterator):

    def __init__(self, total, selection):
        assert total != 0, "Cannot create FixedRange on empty set"
        if not selection:
            self.selection = range(total)
        else:
            selection = np.array(selection)
            self.selection = selection[selection < total]

    def get_batch_count(self):
        return 1

    def get_indices(self, batch_index):
        return self.selection


