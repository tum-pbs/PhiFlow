import sys
try:
    import phi
except ImportError:
    print("phiflow is not installed. Visit https://tum-pbs.github.io/PhiFlow/Installation_Instructions.html for more information."
          "\nrun 'pip install phiflow' to install the latest stable version or add the phiflow source directory to your Python PATH.", file=sys.stderr)
    exit(1)

from phi._troubleshoot import assert_minimal_config, troubleshoot
try:
    assert_minimal_config()
except AssertionError as fail_err:
    print("\n".join(fail_err.args), file=sys.stderr)
    exit(1)
print(f"\nInstallation verified. {troubleshoot()}")
