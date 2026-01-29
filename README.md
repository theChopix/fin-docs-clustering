# Financial Documents Clustering

## Development Setup

### Makefile

This project uses a simple `Makefile` utilizing `pip-tools` to manage Python dependencies in a **local virtual environment**. 

#### Makefile commands

```bash
make install_dev
```
Create the `.venv` virtual environment, install `pip-tools`, and install dependencies from `requirements.txt`.

```bash
make compile
```
Compiles dependencies from `pyproject.toml` into a pinned `requirements.txt` using `pip-tools`.

```bash
make sync
```
Synchronizes your virtual environment with the exact versions listed in `requirements.txt`.

```bash
make up
```
Runs both `compile` and `sync` - updating and installing dependencies in one step.

### Pre-commit Setup

This project uses **pre-commit** to run code quality checks such as ruff and mypy before each commit.

After installing the dependencies from `requirements.txt`, install pre-commit (only needed once per clone):

```bash
pre-commit install