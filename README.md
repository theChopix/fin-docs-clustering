# Financial Documents Clustering

## Overview

This project provides a tool for clustering financial document images ([link](https://www.kaggle.com/datasets/mehaksingal/personal-financial-dataset-for-india)) with configurable feature importance.

It can be executed as a command-line application after installing all dependencies and setting up the required `.env` variables (see below).

```bash
python main.py --input_data_path data --config configs/config.yaml --output_data_path results
```

The input parameters are:

- The path to the input data (or a directory where the data should be downloaded)
- The path to the configuration file that defines the importance (weights) of individual document features (an examplatory one is provided in `configs/config.yaml`)
- The path to the file where the clustering (and intermediate) results will be stored

The overall clustering pipeline consists of three main steps:

1. Feature extraction  
2. Feature engineering  
3. Document clustering  

Each step is described in detail below.

---

### Feature Extraction

In the first step, relevant information is extracted from financial documents (such as bank statements, checks, or salary slips) using a large language model of the vision-language (VLM) type.

The VLM is instructed to extract the following structured features:

```json
{
  "doc_type": "...",
  "issuer": "...",
  "account_present": true/false,
  "amounts": [numbers],
  "dates": ["YYYY-MM-DD"],
  "layout": "table/form/text/mixed",
  "language": "en/hi/other",
  "summary": "one sentence description"
}
```

Where:

- `amounts` contains all monetary values found in the document  
- `dates` contains all detected dates  
- `summary` is a short textual description of the document  

During development and experimentation, the `gpt-5-nano` model from the OpenAI family was used for this task.

---

### Feature Engineering

After feature extraction, the selected features are transformed into numerical representations suitable for clustering. This step includes encoding, normalization, and scaling.

The following transformations are applied:

#### Text Features

- `doc_type`, `issuer`  
  → Converted into vector embeddings using an embedding model  
  → `text-embedding-3-small` was used  
  → Embeddings are L2-normalized

#### Numeric Features

- `amounts`  
  → The maximum value is selected (as it usually represents the most relevant amount)  
  → The value is scaled across the dataset

- `dates`  
  → The latest date is selected (as it is typically the most representative)  
  → Converted to a timestamp  
  → Scaled across the dataset

#### Categorical Features

- `language`  
  → Encoded using one-hot representation  
  → This is suitable because only a limited number of languages is expected

#### Feature Vector Construction

Each transformed feature produces a vector representation (with `amounts` and `dates` resulting in vectors of length 1).

All partial vectors are then concatenated into a single feature vector representing the document.

After concatenation, each feature block is multiplied by a scaling constant defined in the configuration file.

These constants represent the relative importance of individual features in the clustering process.

Although not strictly required by the implementation, it is recommended that all scaling values sum to 1.

The intuition is simple:

> The higher the weight of a feature, the greater its influence on the final clustering result.

---

### Document Clustering

In the final step, a clustering algorithm is applied to the resulting feature vectors.

Currently, the K-Means algorithm is used to group similar documents into clusters based on their combined feature representations.

The output of this step is a cluster assignment for each processed document.


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
