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
```

### Execution

To run the clustering process, execute the following command:

```bash
python main.py --input_data_path data --config configs/config.yaml --output_data_path results
```

This will process the documents in the `data` directory and save the clustering results to the `results` directory.

### Testing

Test modules are located in the `test` directory. To run all tests, use the following command:

```bash
PYTHONPATH=. pytest -v
```
