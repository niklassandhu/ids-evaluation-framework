![Python](https://img.shields.io/badge/python-3.13%2B-blue?logo=python&logoColor=white)
# IDS Evaluation Framework

A comprehensive, modular, and configurable framework for evaluating Machine Learning-based Intrusion Detection Systems (IDS).

![evaluation_pipeline](assets/evaluation_pipeline.png)

- [IDS Evaluation Framework](#ids-evaluation-framework)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Native Installation](#native-installation)
    - [Docker Installation](#docker-installation)
  - [Quick Start](#quick-start)
    - [1. Create a Configuration File](#1-create-a-configuration-file)
    - [2. Prepare Your Data](#2-prepare-your-data)
    - [3. Run Evaluation](#3-run-evaluation)
  - [Usage](#usage)
    - [CLI Commands](#cli-commands)
    - [Evaluation Flags](#evaluation-flags)
    - [Makefile Targets](#makefile-targets)
  - [Configuration](#configuration)
    - [Key Configuration Sections](#key-configuration-sections)
  - [Output Structure](#output-structure)
  - [Plugin Development](#plugin-development)
  - [Development](#development)
  - [Tests](#tests)
  - [License](#license)
  - [Programmier Praktikum](#programmier-praktikum)

## Features

- **Modular Plugin Architecture**: Easily extend the framework with custom IDS models, metrics, and adversarial attacks
- **Flexible Data Pipeline**: Load, preprocess, and split datasets with configurable preprocessing steps and feature selection
- **Multiple Evaluation Modes**: Support for intra-dataset, cross-dataset, and k-fold cross-validation evaluation
- **Comprehensive Metrics**: Built-in static metrics (accuracy, F1, precision, recall, ROC-AUC, etc.) and runtime metrics (CPU, RAM, training time)
- **Adversarial Robustness Testing**: Evaluate model robustness against adversarial attacks (FGSM, noise perturbation, junk data injection)
- **Reproducible Results**: Hash-based output organization ensures consistent experiment tracking
- **Flexible Deployment**: Run natively with Python or via Docker

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip

### Native Installation

```bash
# Install dependencies (uv should be in your $PATH)
uv sync

# Verify installation
uv run ids-eval version
```

### Docker Installation

```bash
# Configure environment variables
cp .env.example .env
# Edit .env to set your data paths

# Build and run via Docker Compose
docker compose build
docker compose run --rm ids-eval version
```

The image is built locally from the provided `Dockerfile`.

## Quick Start

### 1. Create a Configuration File

Copy the example configuration and adjust it to your needs:

```bash
cp examples/run_config/example.config.yml examples/run_config/my_config.yml
```

### 2. Prepare Your Data

Run the data preparation pipeline:

```bash
uv run ids-eval dataset <run_config>
```

### 3. Run Evaluation

Execute the evaluation pipeline:

```bash
uv run ids-eval evaluate <run_config>
```

## Usage

### CLI Commands

The framework provides two main commands:

| Command | Description                |
|---------|----------------------------|
| `ids-eval dataset <config.yml>` | Run dataset pipeline       |
| `ids-eval evaluate <config.yml>` | Run evaluation pipeline |

### Evaluation Flags

| Flag                  | Description |
|-----------------------|-------------|
| `--train-only`        | Only train models, skip testing phase |
| `--force-train`       | Force retraining, ignore saved models |
| `--force-model`       | Load saved models without config hash validation |
| `--clear-checkpoints` | Clear evaluation checkpoints before running |

### Makefile Targets

```bash
make dataset CONFIG=<config.yml>          # Run dataset pipeline
make evaluate CONFIG=<config.yml>         # Run evaluation pipeline
make docker-dataset CONFIG=<config.yml>   # Run dataset pipeline via Docker
make docker-evaluate CONFIG=<config.yml>  # Run evaluation via Docker
make help                                 # Show all available targets
```

## Configuration

The framework uses YAML configuration files. See `examples/run_config/example.config.yml` for a fully documented example.

### Key Configuration Sections

- **general**: Run name, paths, random seed
- **data_manager**: Dataset loading, preprocessing, feature selection, train/test split
- **evaluation**: IDS models, metrics, adversarial attacks

## Output Structure

All outputs are organized in hash-based directories for reproducibility:

```
out/
├── processed_datasets/<hash>/    # Preprocessed datasets
├── saved_models/<hash>/          # Trained models
└── reports/<hash>/               # Evaluation reports
    ├── config.yaml               # Configuration used
    ├── dataset_report.yaml       # Dataset statistics
    ├── ids_report.yaml           # Detailed evaluation results
    └── evaluation_summary.yaml   # Aggregated summary
```

The configuration hash is displayed at startup:
```
Your config hash is: a1b2c3d4
```

## Plugin Development

The framework supports four types of plugins:

| Plugin Type | Directory | Base Class |
|-------------|-----------|------------|
| IDS Models | `plugin_ids/` | `AbstractIDSConnector` |
| Static Metrics | `plugin_static_metric/` | `AbstractStaticMetric` |
| Runtime Metrics | `plugin_runtime_metric/` | `AbstractRuntimeMetric` |
| Adversarial Attacks | `plugin_adversarial/` | `AbstractAdversarialAttack` |

See the existing plugins in each directory for implementation examples.

## Development

```bash
make setup      # Install dependencies
make test       # Run tests
make lint       # Check code style
make format     # Format code
```

## Tests

The test suite lives in `tests/` and is run with pytest:

```bash
uv run pytest          # run all tests
make test              # equivalent target
```

`tests/test_metrics.py` contains two tests that verify the mathematical
correctness of two static metrics:

| Test | Checks |
|------|--------|
| `test_pr_auc_average_precision` | PR-AUC (Average Precision) against a known reference value |
| `test_robustness_index_normalized_area` | Robustness Index equals the normalized area under the accuracy–perturbation curve |

Run a single test:

```bash
uv run pytest tests/test_metrics.py -k pr_auc
```

## BibTeX entry
Please cite this project using the following bibtex entry: <br>
[![Generic badge](https://img.shields.io/badge/Peer%20Reviewed-No-red.svg)](https://shields.io/)
```bibtex
@inproceedings{}
```
## License

See [LICENSE](LICENSE) for details.

## Programmier Praktikum

This project was extended in the Programmier-Praktikum with a raw-pcap ingestion path
(nfstream) and two metrics, the **PR-AUC** and the **Robustness
Index (RI)** demonstrated on the **Apollon** MAB IDS.

**Architecture** A layered plugin pipeline setup entirely by a YAML run-config:
`dataset_pipeline` builds the dataset(s) (CSV, or pcap via NFStream + time-window labeling),
`evaluation_pipeline` trains the IDS plugins and runs the adversarial/robustness sweep,
`metrics_pipeline` computes the metrics. IDS models, metrics and attacks are swappable
plugins resolved through the `registry` via an abstract class.

Two example runs are provided and both evaluate **Apollon** (attacked via a trained surrogate)
and report PR-AUC, the RI and further metrics under an FGSM sweep.

**1. pcap run (single day).**
[`examples/run_config/PP_EXAMPLE.yml`](examples/run_config/PP_EXAMPLE.yml) ingests the
CICIDS2017 **Friday** capture directly from raw pcap via nfstream (+ time-window labeling):

```bash
uv run ids-eval dataset  examples/run_config/PP_EXAMPLE.yml
uv run ids-eval evaluate examples/run_config/PP_EXAMPLE.yml
```

**2. full-dataset run (all days, CSV).**
[`examples/run_config/PP_EXAMPLE_full.yml`](examples/run_config/PP_EXAMPLE_full.yml) runs on
the **complete** CICIDS2017, **no** nfstream. The much larger
sample gives more robust results than the friday pcap run:

```bash
uv run ids-eval dataset  examples/run_config/PP_EXAMPLE_full.yml
uv run ids-eval evaluate examples/run_config/PP_EXAMPLE_full.yml
```

For further information to correctly configure the run please refer to
[`examples/run_config/example.config.yml`](examples/run_config/example.config.yml).

**Dataset.**
- For run **1** download `Friday-WorkingHours.pcap`.
- For run **2** download the full **`MachineLearningCVE`** / `GeneratedLabelledFlows` CSV set
  (the whole labeled CICIDS2017).

Both are available from the CIC-IDS2017 dataset
(<https://www.unb.ca/cic/datasets/ids-2017.html>). Place the files in
`raw_data/cic_ids_2017_flow/`, matching the `base_path`/`subpath` in each config.
