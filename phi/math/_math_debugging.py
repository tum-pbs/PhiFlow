import sys
import gc

from ._tensors import Tensor


def count_tensors_in_memory(min_print_size: int = None):
    gc.collect()
    total = 0
    bytes = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Tensor):
                total += 1
                size = obj.shape.volume * obj.dtype.itemsize
                bytes += size
                if isinstance(min_print_size, int) and size >= min_print_size:
                    print(f"Tensor '{obj}' ({sys.getrefcount(obj)} references)")
                    # referrers = gc.get_referrers(obj)
                    # print([type(r) for r in referrers])
        except Exception:
            pass
    print(f"There are {total} Φ-Tensors with a total size of {bytes / 1024 / 1024:.1f} MB")
