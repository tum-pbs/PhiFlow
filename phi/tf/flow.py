from phi._flow import *
from phi.tf.profiling import Timeliner
import tensorflow as tf
import os
from tensorflow.python.client import device_lib


class TFFluidSimulation(FluidSimulation):

    def __init__(self, shape, boundary='open', batch_size=None, session=None, solver=None, **kwargs):
        math.load_tensorflow()
        # Init
        self.session = session if session else tf.Session()
        self.graph = tf.get_default_graph()
        gpus = [device.name for device in device_lib.list_local_devices() if device.device_type == 'GPU']
        if solver is None and len(gpus) > 0:
            from phi.solver.cuda import CUDA
            solver = CUDA()
        FluidSimulation.__init__(self, shape, boundary=boundary, batch_size=batch_size, solver=solver, **kwargs)
        self.timeliner = None
        self.timeline_file = None
        self.summary_writers = {}
        self.summary_directory = ''

        if self.timeliner:
            self.timeliner.add_run()


    def placeholder(self, element_type=1, name=None, batch_size=None, dtype=np.float32):
        import tensorflow as tf
        if element_type == "velocity":
            element_type = "staggered" if self._mac else "vector"
        array = tf.placeholder(dtype, self.shape(element_type, batch_size), name=name)
        if element_type == "staggered":
            return StaggeredGrid(array)
        else:
            return array

    def clear_domain(self):
        # Active / Fluid Mask
        if self._force_use_masks or self._active_mask is not None or self._fluid_mask is not None:
            self._active_mask = self._create_or_reset_mask(self._active_mask)
            self._fluid_mask = self._create_or_reset_mask(self._fluid_mask)
        # Velocity Mask
        if self._force_use_masks or self._active_mask is not None or self._fluid_mask is not None:
            self._update_velocity_mask()

    def set_obstacle(self, mask_or_size, origin=None):
        if self._active_mask is None:
            self._active_mask = self._create_or_reset_mask(None)
        if self._fluid_mask is None:
            self._fluid_mask = self._create_or_reset_mask(None)

        dims = range(self.rank)

        if isinstance(mask_or_size, np.ndarray):
            value = mask_or_size
            slices = None
            raise NotImplementedError() # TODO
        else:
            # mask_or_size = tuple/list of extents
            if isinstance(mask_or_size, int):
                mask_or_size = [mask_or_size for i in dims]
            if origin is None:
                origin = [0 for i in range(len(mask_or_size))]
            else:
                origin = list(origin)
            fluid_mask_data, active_mask_data = self.session.run([self._fluid_mask, self._active_mask])
            fluid_mask_data[[0]+[slice(origin[i], origin[i]+mask_or_size[i]) for i in dims]+[0]] = 0
            active_mask_data[[0]+[slice(origin[i], origin[i]+mask_or_size[i]) for i in dims]+[0]] = 0
            self.session.run([self._fluid_mask.assign(fluid_mask_data), self._active_mask.assign(active_mask_data)])
        self._update_velocity_mask()

    def _create_or_reset_mask(self, old_mask):
        if old_mask is None:
            return tf.Variable(self._create_mask(), dtype=tf.float32)
        else:
            self.session.run(old_mask.assign(self._create_mask()))
            return old_mask

    def _update_velocity_mask(self):
        new_velocity_mask = self._boundary.create_velocity_mask(self._fluid_mask, self._dimensions, self._mac)
        if self._velocity_mask is None:
            self._velocity_mask = StaggeredGrid(tf.Variable(new_velocity_mask.staggered, dtype=tf.float32))
        else:
            self.session.run(self._velocity_mask.staggered.assign(new_velocity_mask.staggered))

    def _update_masks(self, cell_type_mask):
        if self.cell_type_mask is None:
            self.cell_type_mask = tf.Variable(cell_type_mask, dtype=tf.float32)
        else:
            self.session.run(self.cell_type_mask.assign(cell_type_mask))

        new_velocity_mask = self.boundary.create_velocity_mask(cell_type_mask, self._mac)
        if new_velocity_mask is None:
            self.velocity_mask = None
        else:
            if self.velocity_mask is None:
                self.velocity_mask = StaggeredGrid(tf.Variable(new_velocity_mask.staggered, dtype=tf.float32))
            else:
                self.session.run(self.velocity_mask.staggered.assign(new_velocity_mask.staggered))





def placeholder(element_type=1, name=None, batch_size=None, dtype=np.float32):
    return _default_phi_stack.get_default().placeholder(element_type, name=name, batch_size=batch_size, dtype=dtype)


def run(tasks, feed_dict=None, options=None, run_metadata=None, summary_key=None, time=None, merged_summary=None):
    return _default_phi_stack.get_default().run(tasks, feed_dict, options, run_metadata, summary_key=summary_key,
                                                time=time, merged_summary=merged_summary)