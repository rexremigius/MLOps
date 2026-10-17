# Changes made for the Lab2 assignment
 
`src/data.py` (new)
 
* Loads the Pima Indians diabetes dataset from OpenML (768 patients, 8 features). The target is yes/no: does the patient have diabetes.
* Splits it 80/20 with a fixed seed so every script uses the same test set.
* Caches the download in `data/`.
`src/train_model.py`
 
* Switched from random `make_classification` data to the diabetes dataset. The old code could also pick `n_samples=0`.
* Switched from a random forest to gradient boosting.
* Saves the model straight into `models/` instead of the working directory.
* MLflow now logs to `sqlite:///mlflow.db`, since newer MLflow versions no longer accept the `./mlruns` folder.
* Switched file paths to `pathlib` and removed unused imports.
`src/evaluate_model.py`
 
* Fixed the evaluation. It used to generate a new random dataset, so the scores didn't mean anything. It now uses the held-out test set.
* Saves accuracy, F1 and ROC AUC to `metrics/`.
* Added a quality gate: if ROC AUC is below 0.75, the script fails and nothing gets committed. I went with AUC over F1 because only about 35% of patients are positive.
`src/calibrate_model.py` (new)
 
* The workflow is called "calibration", but the original code never calibrated anything. This script wraps the model in `CalibratedClassifierCV` (isotonic by default, `--method sigmoid` for Platt).
* Picks the latest model automatically and saves the Brier score before and after.
`src/test.ipynb`
 
* Rewritten for the new dataset and model.
* Fixed the old bugs: `toarry`, swapped `pickle.dump` arguments, `test_size=0.9`, `.jopblib`.
`model_calibration_on_push.yml`
 
* Retrains only when Lab2 changes, not on every push to `main`.
* Train, evaluate, upload the MLflow db as an artifact, commit model and metrics.
`model_calibration.yml`
 
* Now actually calibrates the latest model (before, it just retrained again).
* Runs nightly, after every successful retrain, and on manual trigger.
* Fixed `$metrics_filename` being undefined in the commit step.
Both workflows
 
* Upgraded to `checkout@v4`, `setup-python@v5` and Python 3.11, with `cache: pip`.
* Removed the file permission steps and the `mv` steps, since neither is needed anymore.
* Commits come from the GitHub Actions bot instead of a personal account.
* Added `git pull --rebase` before pushing.
## Results (local run)
 
Accuracy 0.747, F1 0.621, ROC AUC 0.827. Brier score went from 0.1647 to 0.1604 after isotonic calibration (sigmoid gave 0.1609).
 
## Run locally
 
```bash
pip install -r requirements.txt
python src/train_model.py --timestamp local
python src/evaluate_model.py --timestamp local
python src/calibrate_model.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
 
---


# Using GitHub Actions for Model Training and Versioning

This repository demonstrates how to use GitHub Actions to automate the process of training a machine learning model, storing the model, and versioning it. This allows you to easily update and improve your model in a collaborative environment.

Watch the tutorial video for this lab at [Github action Lab2](https://youtu.be/cj5sXIMZUjQ)


## Prerequisites

- [GitHub](https://github.com) account
- Basic knowledge of Python and machine learning
- Git command-line tool (optional)

## Getting Started

1. **Fork this Repository**: Click the "Fork" button at the top right of this [repository](https://github.com/raminmohammadi/MLOps/) to create your own copy.
3. **Clone Your Repository**:
   ```bash
   git clone https://github.com/your-username/your-forked-repo.git
   cd your-forked-repo

   ```
4. GitHub account
5. Basic knowledge of Python and machine learning
6. Git command-line tool (optional)

# Running the Workflow
## Customize Model Training
1. Modify the `train_model.py` script in the `src/` directory according to your dataset and model requirements. This script generates synthetic data for demonstration purposes.

## Push Your Changes:
1. Commit your changes and push them to your forked repository.

## GitHub Actions Workflow:
1. Once you push changes to the main branch, the GitHub Actions workflow will be triggered automatically.

## View Workflow Progress:
1. You can track the progress of the workflow by going to the "Actions" tab in your GitHub repository.

## Retrieve the Trained Model:
1. After the workflow completes successfully, the trained model will be stored in the `models/` directory.

# Model Evaluation
The model evaluation is performed automatically within the GitHub Actions workflow. The evaluation results (e.g., F1 Score) are stored in the `metrics/` directory.

# Versioning the Model
Each time you run the workflow, a new version of the model is created and stored. You can access and use these models for your projects.

# GitHub Actions Workflow Details
The workflow consists of the following steps:

- Generate and Store Timestamp: A timestamp is generated and stored in a file for versioning.
- Model Training: The `train_model.py` script is executed, which trains a random forest classifier on synthetic data and stores the model in the `models/` directory.
- Model Evaluation: The `evaluate_model.py` script is executed to evaluate the model's F1 Score on synthetic data, and the results are stored in the `metrics/` directory.
- Store and Version the New Model: The trained model is moved to the `models/` directory with a timestamp-based version.
- Commit and Push Changes: The metrics and updated model are committed to the repository, allowing you to track changes.

# Model Calibration Workflow
## Overview
The `model_calibration_on_push.yml` workflow is a part of the automation process for machine learning model calibration within this repository. It is essential for ensuring that the model's predictions are accurate and well-calibrated, a critical step in machine learning applications.

## Workflow Purpose
This workflow's primary purpose is to calibrate a trained machine learning model after each push to the main branch of the repository. Calibration is a crucial step to align model predictions with reality, particularly when dealing with classification tasks. In simple terms, calibration ensures that a model's predicted probabilities match the actual likelihood of an event happening.

## Workflow Execution
Let's break down how this workflow operates step by step:

### Step 1: Trigger on Push to Main Branch
- This workflow is automatically initiated when changes are pushed to the main branch of the repository. It ensures that the model remains calibrated and up-to-date with the latest data and adjustments.

### Step 2: Prepare Environment
- The workflow begins by setting up a Python environment and installing the necessary Python libraries and dependencies. This is crucial to ensure that the model calibration process can be executed without any issues.

### Step 3: Load Trained Model
- The trained machine learning model, which has been previously saved in the `models/` directory, is loaded into memory. This model should be the most recent version, as trained by the `train_model.py` script.

### Step 4: Calibrate Model Probabilities
- In this step, the model's predicted probabilities are calibrated. Calibration methods, such as Platt scaling or isotonic regression, are applied. These methods adjust the model's predicted probabilities to match the actual likelihood of an event occurring. This calibration step is critical for reliable decision-making based on the model's predictions.

### Step 5: Save Calibrated Model
- The calibrated model is saved back to the `models/` directory. It is given a distinct identifier to differentiate it from the original, uncalibrated models. This ensures that both the original model and the calibrated model are available for comparison and use.

### Step 6: Commit and Push Changes
- This final step involves committing the calibrated model and any other relevant files to the repository. It is essential to keep track of the changes made during the calibration process and to store the calibrated model in the repository for future applications and reference.

# Customization
The `model_calibration_on_push.yml` workflow can be customized to align with your specific project requirements. You can modify calibration methods, the directory where the calibrated model is saved, or any other aspects of the calibration process to meet your project's unique needs.

# Integration with Model Training
This workflow is designed to work seamlessly with the main model training workflow, `model_retraining_on_push.yml`. In the initial workflow, the model is trained, and in this workflow, the calibrated model is generated. The calibrated model can then be used in applications where precise, well-calibrated probabilities are essential.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
- This project uses GitHub Actions for continuous integration and deployment.
- Model training and evaluation are powered by Python and scikit-learn.

# Questions or Issues
If you have any questions or encounter issues while using this GitHub Actions workflow, please open an issue in the Issues section of your repository.

