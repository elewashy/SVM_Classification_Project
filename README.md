# SVM Classification from Scratch on Titanic Dataset

This project implements a Support Vector Machine (SVM) classifier from scratch using Python. The classifier is trained and evaluated on the classic Titanic dataset to predict passenger survival.

## Project Overview

The `SVM_Classification.ipynb` Jupyter Notebook contains the complete workflow:

1.  **Data Loading:** Loads the Titanic dataset (`Titanic-Dataset.csv`).
2.  **Exploratory Data Analysis (EDA):** Initial exploration and visualization of the data using libraries like Pandas, Matplotlib, and Seaborn.
3.  **Data Preprocessing:** Handles missing values, performs feature engineering (dropping irrelevant columns, converting categorical features), and scales numerical features.
4.  **SVM Implementation:** Defines a Python class `SVM` that implements the classifier logic, including the hinge loss function and gradient descent for training.
5.  **Model Training:** Instantiates and trains the custom SVM classifier on the preprocessed training data.
6.  **Model Evaluation:** Evaluates the trained model on the test set using metrics like accuracy, confusion matrix, precision, recall, and F1-score. Includes visualization of the loss curve and confusion matrix.

## Dataset

The project uses the [Titanic dataset](https://www.kaggle.com/c/titanic/data), which is included in this repository as `Titanic-Dataset.csv`.

## Requirements

The necessary Python libraries are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

## How to Run

1.  Ensure you have Python and pip installed.
2.  Clone this repository or download the files.
3.  Install the required libraries: `pip install -r requirements.txt`
4.  Open and run the `SVM_Classification.ipynb` notebook using Jupyter Notebook or JupyterLab.

```bash
jupyter notebook SVM_Classification.ipynb
# or
jupyter lab SVM_Classification.ipynb
