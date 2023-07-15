from contextlib import contextmanager
from os.path import dirname

import packaging.version


def assert_minimal_config():  # raises AssertionError
    import sys
    assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "phiflow requires Python 3.6 or newer to run"

    try:
        import numpy
    except ImportError:
        raise AssertionError("phiflow is unable to run because NumPy is not installed.")
    try:
        import scipy
    except ImportError:
        raise AssertionError("phiflow is unable to run because SciPy is not installed.")
    from phi import flow
    from phi import math
    with math.NUMPY:
        a = math.ones()
        math.assert_close(a + a, 2)


def troubleshoot():
    import phi
    return f"PhiFlow {phi.__version__} at {dirname(__file__)}\n"\
           f"Web interface: {troubleshoot_dash()}\n"


def troubleshoot_dash():
    try:
        import dash
    except ImportError:
        return "Dash not installed. Will fallback to console interface. To install dash, run  $ pip install dash"
    try:
        import plotly
    except ImportError:
        return "Plotly not installed. This package is required by dash. To install it, run  $ pip install plotly"
    try:
        dash.Dash('Test')
    except BaseException as e:
        return f"Dash ({dash.__version__}) runtime error: {e}"
    return f"OK (dash {dash.__version__})"
