# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset

# Overview
This project implements a binary classification model to predict whether an individual earns more than $50K annually based on census data. The dataset used is the UCI Census Income (Adult) dataset. The model is built using PyTorch and incorporates embedding layers for categorical variables and batch normalization for continuous features.

# Features
Data preprocessing with categorical encoding and tensor conversion.

Neural network model supporting embeddings for categorical variables.

Training with Cross-Entropy loss and Adam optimizer.

Evaluation on a held-out test set with accuracy computation.

Interactive user input for real-time income prediction.

Visualization of training loss over epochs.

# Dataset
Source: UCI Machine Learning Repository - Adult Dataset

Number of Records: 30,000 (preprocessed subset with no missing values)

Features: 5 categorical, 2 continuous, 1 binary label

Task: Predict income level (>50K or ≤50K)

Getting Started Prerequisites Python 3.x

PyTorch

pandas

numpy

scikit-learn (for shuffling)

matplotlib

Install dependencies with:

text pip install torch pandas numpy scikit-learn matplotlib Running the Code Clone the repository:

text git clone cd tabular-income-classification-pytorch-census-ml Ensure the dataset CSV (income.csv) is in the Data folder as expected.

Run the notebook or script to train the model and evaluate performance.

Use the interactive prompt to input personal details and get income prediction.

# Code Structure
data_preprocessing.py: Loads and preprocesses the dataset.

model.py: Contains the PyTorch TabularModel class.

train.py: Runs training loops and saves model checkpoints.

evaluate.py: Evaluates model performance on test data.

predict.py: Interactive script to input new data and predict income.

notebook.ipynb: Complete walkthrough including code cells and explanations.

# Results
Typical test accuracy: ~85%

Loss decreases smoothly over 300 epochs.

Embeddings help efficiently model categorical features.

# Future Work
Experiment with deeper architectures and hyperparameters.

Add support for missing values and imbalanced data techniques.

Deployment as an API or web app for broader accessibility.
