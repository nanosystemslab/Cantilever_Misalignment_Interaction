# Cantilever_Misalignment_Interaction
### Code DOI: [![Zenodo-Code-DOI](https://zenodo.org/badge/912981011.svg)](https://doi.org/10.5281/zenodo.14679787)
### Data DOI: [![Zenodo-Data-DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14654525.svg)](https://doi.org/10.5281/zenodo.14654525)

## Overview
**Cantilever_Misalignment_Interaction** is a research project developed to investigate and model the interaction effects of cantilever misalignment in experimental setups. This repository contains the necessary code and resources for simulating, analyzing, and visualizing misalignment interactions using experimental data and computational models.

## Directory Structure

```
Cantilever_Misalignment_Interaction/
├── README.md
├── cantilever_misalignment_interaction
│   ├── init.py
│   ├── exp_and_mod.py
│   └── gen_curves_fit.py
├── data
├── out
├── poetry.lock
├── pyproject.toml
├── ref
└── tests
└── init.py
```

## Features
- **Experimental and Model Analysis**: Compare experimental results with theoretical models of cantilever misalignment interaction.
- **Curve Fitting**: Generate and fit curves to describe observed interaction patterns.
- **Data Visualization**: Create plots to visualize experimental and model data for deeper insights.

## Collect Data
Data for this project can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.14654524). Extract the data files into the `data` directory.

### Data Citation
If you use this data, please cite it using the following BibTeX entry:
```bibtex
@dataset{dataset_mnakamura,
	author = {Matthew, Nakamura and Rocheville, Ethan and Peterson, Kirsten and Heyes, Corissa and Brown, Joseph},
	doi = {10.5281/zenodo.14654525},
	month = jan,
	publisher = {Zenodo},
	title = {Cantilever Misalignment Interaction Dataset},
	url = {https://doi.org/10.5281/zenodo.14654525},
	year = 2025,
	bdsk-url-1 = {https://doi.org/10.5281/zenodo.14654525}}
```

## Installation
### Clone Repository

To clone the repository:

```sh
git clone https://github.com/yourusername/Cantilever_Misalignment_Interaction.git
cd Cantilever_Misalignment_Interaction
```

### Using pip
install dependencies using pip:
```sh
pip install .
```

### Using Poetry

Install dependencies using Poetry:

```sh
poetry install
```

## Usage
	1.	Prepare Input Data: Place your input data files in the data directory.
	2.	Run Scripts: Execute the scripts in the cantilever_misalignment_interaction directory to analyze and generate results.

```sh
python cantilever_misalignment_interaction/exp_and_mod.py
python cantilever_misalignment_interaction/gen_curves_fit.py
```

	3.	View Results: Access output plots and data files in the out directory.

## Results

Figure 1. Experimental vs Model Interaction Curves

TODO: Add a description of the key results and link the relevant output files.

### Code Citation

If you use this code, please cite it using the following BibTeX entry:

```bibtex
@software{matthew_nakamura_2025_14679788,
	author = {Matthew Nakamura},
	doi = {10.5281/zenodo.14679788},
	month = jan,
	publisher = {Zenodo},
	swhid = {swh:1:dir:6559c6f0993854ca6dfab7e3d1df0454470124d3 ;origin=https://doi.org/10.5281/zenodo.14679787;vi sit=swh:1:snp:68a001382c552106c3888bafd1e2158940ce 20f8;anchor=swh:1:rel:26ec33751b594173cfa751b45259 6e5523580d7e;path=nanosystemslab- Cantilever\_Misalignment\_Interaction-11ec5e2},
	title = {nanosystemslab/Cantilever\_Misalignment\_Interaction : Initial Release of Cantilever Misalignment Interaction Data Processing (v1.0.0)},
	url = {https://doi.org/10.5281/zenodo.14679788},
	version = {v1.0.0},
	year = 2025,
	bdsk-url-1 = {https://doi.org/10.5281/zenodo.14679788}}
```

## License

This project is licensed under the GPL-3.0-or-later License. See the LICENSE file for details.

Replace `xxxxxxxx`, `your_name`, and `yourusername` with your actual information.
