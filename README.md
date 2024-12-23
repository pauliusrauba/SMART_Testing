# Context-Aware Testing (CAT) with SMART Testing

This repository implements SMART. 

# Overview
This repository contains scripts, datasets, and experiments to reproduce the findings in our paper. The SMART Testing system automatically identifies more relevant and impactful failures than traditional methods, highlighting the potential of CAT as a new paradigm for ML model testing.

# Prerequisites
To use this repository, you will need an API key for querying the LLM. The simplest way to set this up is to create a Python file named `openai_config.py` in the root directory with the following structure:

```python
def get_openai_config():
    openai_config = {
        "api_type": "azure",
        "api_base": api_base,
        "api_version": api_version,
        "api_key": api_key_main,
        "deployment_id": deployment_name,
        "deployment_id_ada": deployment_name_ada,
        "temperature": 0.0,
        "seed": 0
    }
    return openai_config
```
Replace the placeholders (api_base, api_version, api_key_main, deployment_name, deployment_name_ada) with your actual configuration values. 


# Getting Started
### 1. Install Dependencies
Ensure you have Python 3.10 or later installed. Create a virtual environment and install the necessary packages by running:

``` python -m venv venv source venv/bin/activate # On macOS/Linux venv\Scripts\activate # On Windows pip install -r requirements.txt ```

### 2. Running Experiments
You can reproduce the experiments from the paper using the provided scripts.

Run a specific experiment: To execute one of the experiments, run:

``` python experiments/exp1.py ```

Replace exp1.py with any of the experiment scripts available (exp2.py, exp3.py, exp4.py).

### 3. Example usage
An example how to use SMART is provided under notebooks/Usage Example.ipynb

### 4. Citation
If you found this repository useful, please consider citing our paper:

```
@article{rauba2024context,
  title={Context-Aware Testing: A New Paradigm for Model Testing with Large Language Models},
  author={Rauba, Paulius and Seedat, Nabeel and Luyten, Max Ruiz and van der Schaar, Mihaela},
  journal={Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
