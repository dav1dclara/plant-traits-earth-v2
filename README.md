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
