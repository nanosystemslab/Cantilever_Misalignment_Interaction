Cantilever_Misalignment_Interaction

Code DOI: 

Data DOI: 

Overview

Cantilever_Misalignment_Interaction is a research project developed to investigate and model the interaction effects of cantilever misalignment in experimental setups. This repository contains the necessary code and resources for simulating, analyzing, and visualizing misalignment interactions using experimental data and computational models.

Directory Structure

Cantilever_Misalignment_Interaction/
├── README.md
├── cantilever_misalignment_interaction
│   ├── __init__.py
│   ├── exp_and_mod.py
│   └── gen_curves_fit.py
├── data
├── out
├── poetry.lock
├── pyproject.toml
├── ref
└── tests
    └── __init__.py

Features
	•	Experimental and Model Analysis: Compare experimental results with theoretical models of cantilever misalignment interaction.
	•	Curve Fitting: Generate and fit curves to describe observed interaction patterns.
	•	Data Visualization: Create plots to visualize experimental and model data for deeper insights.

Collect Data

Data for this project can be downloaded from Zenodo. Extract the data files into the data directory.

Data Citation

If you use this data, please cite it using the following BibTeX entry:

@dataset{your_name_2025_xxxxxxxx,
  author       = {Your Name and Collaborators},
  title        = {Cantilever Misalignment Interaction Dataset},
  month        = jan,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.xxxxxxxx},
  url          = {https://doi.org/10.5281/zenodo.xxxxxxxx}
}

Installation

Clone Repository

To clone the repository:

git clone https://github.com/yourusername/Cantilever_Misalignment_Interaction.git
cd Cantilever_Misalignment_Interaction

Using Poetry

Install dependencies using Poetry:

poetry install

Usage
	1.	Prepare Input Data: Place your input data files in the data directory.
	2.	Run Scripts: Execute the scripts in the cantilever_misalignment_interaction directory to analyze and generate results.

python cantilever_misalignment_interaction/exp_and_mod.py
python cantilever_misalignment_interaction/gen_curves_fit.py

	3.	View Results: Access output plots and data files in the out directory.

Results

Figure 1. Experimental vs Model Interaction Curves

TODO: Add a description of the key results and link the relevant output files.

Code Citation

If you use this code, please cite it using the following BibTeX entry:

@software{your_name_2025_xxxxxxxx,
  author       = {Your Name},
  title        = {Cantilever Misalignment Interaction},
  month        = jan,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.xxxxxxxx},
  url          = {https://doi.org/10.5281/zenodo.xxxxxxxx}
}

License

This project is licensed under the GPL-3.0-or-later License. See the LICENSE file for details.

Replace the placeholders (xxxxxxxx, your_name, yourusername) with the actual information relevant to your project. Add figures and descriptions for the results as they become available.
