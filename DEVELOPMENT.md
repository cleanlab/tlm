# Development

This project uses the [Hatch] project manager ([installation instructions][hatch-install]).

Hatch automatically manages dependencies and runs testing, type checking, and other operations in isolated [environments][hatch-environments].

[Hatch]: https://hatch.pypa.io/
[hatch-install]: https://hatch.pypa.io/latest/install/
[hatch-environments]: https://hatch.pypa.io/latest/environment/

## Testing

You can run the tests on your local machine with:

```bash
hatch test
```

The [`test` command][hatch-test] supports options such as `-c` for measuring test coverage, `-a` for testing with a matrix of Python versions, and appending an argument like `tests/test_validator.py::test_validate_expert_answer` for running a single test.

[hatch-test]: https://hatch.pypa.io/latest/tutorials/testing/overview/

## Type checking

You can run the [mypy static type checker][mypy] with:

```bash
hatch run types:check
```

[mypy]: https://mypy-lang.org/

## Formatting and linting

You can run the [Ruff][ruff] formatter and linter with:

```bash
hatch fmt
```

This will automatically make [safe fixes][fix-safety] to your code. If you want to only check your files without making modifications, run `hatch fmt --check`.

[ruff]: https://github.com/astral-sh/ruff
[fix-safety]: https://docs.astral.sh/ruff/linter/#fix-safety

## Pre-commit

You can install the pre-commit hooks to automatically run type checking, formatting, and linting on every commit.

First, install [pre-commit][pre-commit], for example, with [pipx]:

```bash
pipx install pre-commit
```

Then, install the hooks:

```bash
pre-commit install
```

[pre-commit]: https://pre-commit.com/
[pipx]: https://pipx.pypa.io/

## Documentation

You can preview the [MkDocs][mkdocs] site locally at <http://localhost:8000/> with:

```bash
hatch run docs:serve
```

This will live-reload the site as you make changes. You can run `hatch run docs:build` to build the compiled site into the `site/` directory.

[mkdocs]: https://www.mkdocs.org/

## Packaging

You can use [`hatch build`][hatch-build] to create build artifacts, a [source distribution ("sdist")][sdist] and a [built distribution ("wheel")][bdist].

[hatch-build]: https://hatch.pypa.io/latest/build/
[sdist]: https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist
[bdist]: https://packaging.python.org/en/latest/glossary/#term-Built-Distribution

### How to build and install the package locally

You may want to build and install the package locally - for example, to see how your changes affect other Python code in a script or Jupyter notebook.

To do this, you can build the package with `hatch build` and then install it in your local environment with `pip install dist/tlm-<version>-py3-none-any.whl`.

Alternatively, you can use `pip install -e /path/to/tlm` to install the package from your local code. Note that if you make further local changes after that, you may need to reload the module, i.e. `reload(tlm)`, or restart the kernel.

## Continuous integration

Tests, type checking, and formatting/linting are [checked in the CI workflow][ci-workflow].

[ci-workflow]: .github/workflows/ci.yml
