from unittest import TestCase
import importlib


def get_undocumented_wildcards(modulename):
    namespace = importlib.import_module(modulename)
    loc = namespace.__dict__
    undocumented = []
    for key, val in loc.items():
        if (key[0] != "_") and (key not in {"_", "In", "Out", "get_ipython", "exit", "quit", "join", "S", }):
            description = val.__doc__
            if not description:
                undocumented.append(key)
    return undocumented, len(loc.items())


class TestFlow(TestCase):

    def test_phi_flow(self):
        modulename = "phi.flow"
        undocumented, loc_len = get_undocumented_wildcards(modulename)
        undocumented_fraction = len(undocumented) / loc_len
        self.assertLess(undocumented_fraction, 0.25, f"{len(undocumented)/loc_len:.2%} of {modulename} imports undocumented. Missing Docstrings in {len(undocumented)}/{loc_len}:\n- " + "\n- ".join(undocumented))
