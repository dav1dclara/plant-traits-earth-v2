# Better maps of plant functional traits – towards planttraits.earth v2

## Setup

### Repository

Clone the repository:

```bash
git clone git@github.com:dav1dclara/plant-traits-earth-v2.git
cd plant-traits-earth-v2/
```

### Dependencies

Create a conda environment, then install the dependencies:

```bash
conda create -n PTEV2 python=3.12
conda activate PTEV2
pip install -r requirements.txt
pip install -e .
```

Install the pre-commit hooks for automatic code formatting and linting on each commit:

```bash
pre-commit install
```


## Project structure

```bash
.
├── configs
├── data
│   └── inputs_processed
│       ├── eo_data
│       ├── gbif
│       ├── splot
│       ├── README.md
│       └── trait_mapping.json
├── notebooks
│   └── 0-data_exploration
│       ├── 0.1-earth_observation
│       ├── 0.2-gbif
│       └── 0.3-splot
├── scripts
├── src
│   └── __init__.py
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── pyproject.toml
└── requirements.txt
```


## Pipeline

### Pre-processing




### Data splitting



### Data chipping
