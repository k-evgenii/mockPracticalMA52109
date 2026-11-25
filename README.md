# cluster_maker
`cluster_maker` is a ...

## Allowed libraries
The package only uses:

- Python standard library
- NumPy
- pandas
- matplotlib
- SciPy
- scikit-learn

No other third-party libraries are required and/or are permitted.

Dependency versions are pinned exactly in `pyproject.toml` and locked via `uv.lock`, so `uv sync`
will not upgrade them.

## Main features
- ...
- ...

## Package root directory structure
- `cluster_maker/` - Package directory
- `demo/` – example scripts
- `data/` - csv data file used by the example scripts
- `tests/` – basic unit tests using the standard library `unittest`

## Installation

### With uv (recommended)
From the root directory of the project (the directory containing `pyproject.toml`), run:

```bash
uv sync --locked
```

### With pip
```bash
python -m pip install -e .
```

This installs the package in editable mode, making `cluster_maker` available for import. Changes to the Python source files take effect without reinstalling the package, so you can modify the code and re-run tests or demos directly.

Make sure you have activated the intended Python environment before installing the package.

## Notes on pyproject.toml and the *.egg-info directory
This project includes a small file named pyproject.toml.
You do not need to open or edit it. Its only purpose is to tell Python/pip
that this folder is a valid installable package. Without it, the command
`python -m pip install -e .` would fail.

When you run the installation command, pip automatically creates a directory
called something like:

`cluster_maker.egg-info/`

This folder contains package metadata used internally by Python (file lists,
version information, etc.). It is generated automatically and should not be
edited.
