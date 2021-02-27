# Contributing to Φ<sub>Flow</sub>
All contributions are welcome!
You can mail the developers to get in touch.


## Types of contributions we're looking for
We're open to all kind of contributions that improve or extend the Φ<sub>Flow</sub> library.
We especially welcome
- New equations / solvers
- Code optimizations or native (CUDA) implementations.
- Integrations with other computing libraries.
- Bug fixes

Φ<sub>Flow</sub> is a framework, not an application collection.
While we list applications in the [demos](../demos) directory, these should be short and easy to understand.


## How to Contribute
We recommend you to contact the developers before starting your contribution.
There may already be similar internal work or planned changes that would affect how to code the contribution.

To contribute code, fork Φ<sub>Flow</sub> on GitHub, make your changes, and submit a pull request.
Make sure that your contribution passes all tests.

For commits, we use the following tags in the header:
`[ci]`, `[doc]`, `[fix]`, `[dash]`, `[math]`, `[backend]`, `[geom]`, `[field]`, `[physics]`, `[tests]`.

Example commit header: `[doc] Markdown layout fix`


## Style Guide
Style guidelines make the code more uniform and easier to read.
Generally we stick to the Python style guidelines as outlined in [PEP 8](https://www.python.org/dev/peps/pep-0008/), with some minor modifications outlined below.

Have a look at the [Zen](https://en.wikipedia.org/wiki/Zen_of_Python) [of Python](https://www.python.org/dev/peps/pep-0020/) for the philosophy behind the rules.
We would like to add the rule *Concise is better than repetitive.*

We use PyLint for static code analysis with specific configuration files for
[demos](../demos/.pylintrc),
[tests](../tests/.pylintrc) and the
[code base](../phi/.pylintrc).
PyLint is part of the automatic testing pipeline.
The warning log can be viewed online by selecting a `Tests` job and expanding the pylint output.

### Docstrings
The [API documentation](https://tum-pbs.github.io/PhiFlow/) for Φ<sub>Flow</sub> is generated using [pdoc](https://pdoc3.github.io/pdoc/).
We use [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
with Markdown formatting.

```python
"""
Function description.

*Note in italic.*

Example:
    
    ```python
    def python_example()
    ```

Args:
    arg1: Description.
        Indentation for multi-line description.

Returns:
    Single output. For multi-output use same format as for args.
"""
```

Docstrings for variables are located directly below the public declaration.
```python
variable: type = value
""" Docstring for the variable. """
```


### Additional style choices
- **No line length limit**; long lines are allowed as long as the code is easy to understand.
- **Code comments** should only describe information that is not obvious from the code. They should be used sparingly as the code should be understandable by itself. For documentation, use docstrings instead. Code comments that explain a single line of code should go in the same line as the code they refer to, if possible.
- Code comments that describe multiple lines precede the block and have the format `# --- Comment ---`.
- Avoid empty lines inside of methods. To separate code blocks use multi-line comments as described above.
- Use the apostrophe character ' to enclose strings that affect the program / are not displayed to the user. Use quotes for text such as warnings.
