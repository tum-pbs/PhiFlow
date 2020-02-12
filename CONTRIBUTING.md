# Contributing to Φ<sub>*Flow*</sub>

All contributions are welcome!
You can mail the developers to get in touch.

## Types of contributions we're looking for

We're open to all kind of contributions that improve or extend the Φ<sub>*Flow*</sub> library.
Have a look at the [roadmap](https://github.com/tum-pbs/PhiFlow/projects/1) to see what is planned and what's currently being done.

We especially welcome
- New equations / solvers
- Code optimizations or native (CUDA) implementations.
- Integrations with other computing libraries such as [PyTorch](https://pytorch.org/) or [Jax](https://github.com/google/jax).
- Bug fixes

Φ<sub>*Flow*</sub> is a framework, not an application collection.
While we list applications in the [demos](../demos) directory, these should be short and easy to understand.

## How to Contribute

We recommend you to contact the developers before starting your contribution.
There may already be similar internal work or planned changes that would affect how to code the contribution.
Also check the [roadmap](https://github.com/tum-pbs/PhiFlow/projects/1).

To contribute code, fork Φ<sub>*Flow*</sub> on GitHub, make your changes, and submit a pull request.
Make sure that your contribution passes all tests.

The code you contribute should be able to run in at least 1D, 2D and 3D without additional modifications required by the user.

## Style Guide
Style guidelines make the code more uniform and easier to read.
Generally we stick to the Python style guidelines as outlined in [PEP 8](https://www.python.org/dev/peps/pep-0008/), with some minor modifications outlined below.

Have a look at the [Zen](https://en.wikipedia.org/wiki/Zen_of_Python) [of Python](https://www.python.org/dev/peps/pep-0020/) for the philosophy behind the rules.
We would like to add the rule *Concise is better than repetitive.*

We use PyLint for static code analysis with specific configuration files for
[demos](../demos/.pylintrc),
[tests](../tests/.pylintrc) and the
[code base](../phi/.pylintrc).
PyLint is part of the automatic testing pipeline on [Travis CI](https://travis-ci.com/tum-pbs/PhiFlow). The warning log can be viewed online by selecting a Python 3.6 job on Travis CI and expanding the pylint output at the bottom.

Additional style choices
- **No line length limit**; long lines are allowed.
- **Code comments** should only describe information that is not obvious from the code. They should be used sparingly as the code should be understandable by itself. For documentation, use docstrings instead. Code comments that explain a single line of code should go in the same line as the code they refer to, if possible.
- Code comments that describe multiple lines precede the block and have the format `# --- Comment ---`.
- No empty lines inside of methods. To separate code blocks use multi-line comments as described above.
- Use the apostrophe character ' to enclose strings that affect the program / are not displayed to the user.
- **Sphinx Docstring** format is used throughout the code base
