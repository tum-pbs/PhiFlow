import logging
import sys
from os.path import isfile

import numpy as np

from phi import math
from phi.field import Scene


class SceneLog:

    def __init__(self, scene: Scene):
        self.scene = scene
        self._scalars = {}  # name -> (frame, value)
        self._scalar_streams = {}
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        self.logger = logging.Logger("vis", logging.DEBUG)
        console_handler = self.console_handler = logging.StreamHandler(sys.stdout)
        log_formatter = logging.Formatter("%(message)s (%(levelname)s), %(asctime)sn\n")
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        if self.scene is not None:
            if not isfile(self.scene.subpath("info.log")):
                log_file = self.scene.subpath("info.log")
            else:
                index = 2
                while True:
                    log_file = self.scene.subpath("info_%d.log" % index)
                    if not isfile(log_file):
                        break
                    else:
                        index += 1
            self.log_file = log_file
            file_handler = self.file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(log_formatter)
            self.logger.addHandler(file_handler)
        else:
            self.log_file = None

    def log(self, message):
        self.logger.info(message)

    def log_scalars(self, frame: int, **values: float or math.Tensor):
        """
        Adds `values` to the curves by name.
        This can be used to log the evolution of scalar quantities or summaries.

        The values are stored in a text file within the scene directory.
        The curves may also be directly viewed in the user interface.

        Args:
            frame: step
            values: Values and names to append to the curves, must be numbers or `phi.math.Tensor`.
                If a curve does not yet exists, a new one is created.
        """
        for name, value in values.items():
            assert isinstance(name, str)
            value = float(math.mean(value).mean)
            if name not in self._scalars:
                self._scalars[name] = []
                if self.scene is not None:
                    path = self.scene.subpath(f"log_{name}.txt")
                    self._scalar_streams[name] = open(path, "w")
            self._scalars[name].append((frame, value))
            if self.scene is not None:
                self._scalar_streams[name].write(f"{frame} {value}\n")
                self._scalar_streams[name].flush()

    def get_scalar_curve(self, name) -> tuple:
        frames = np.array([item[0] for item in self._scalars[name]])
        values = np.array([item[1] for item in self._scalars[name]])
        return frames, values

    @property
    def scalar_curve_names(self) -> tuple:
        return tuple(self._scalars.keys())
