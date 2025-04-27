# SVM Classification on Titanic Dataset: From Scratch vs. Scikit-learn

This project explores the implementation and application of Support Vector Machine (SVM) classification for predicting passenger survival on the classic Titanic dataset. It features two approaches:

1.  **SVM from Scratch:** A custom implementation built using Python and NumPy to demonstrate the underlying mechanics of the algorithm, including gradient descent optimization and hinge loss.
2.  **SVM using Scikit-learn:** Utilization of the robust and optimized `SVC` (Support Vector Classifier) from the popular `scikit-learn` library for comparison and benchmarking.

The entire workflow, from data loading to model evaluation, is documented within the `SVM_Classification.ipynb` Jupyter Notebook.

## Project Workflow (`SVM_Classification.ipynb`)

1.  **Import Libraries:** Essential libraries like NumPy, Pandas, Matplotlib, and Seaborn are imported.
2.  **Load Data:** The Titanic dataset (`Titanic-Dataset.csv`) is loaded using Pandas.
3.  **Exploratory Data Analysis (EDA):**
    *   Initial inspection of data structure (`.info()`, `.describe()`).
    *   Checking for missing values (`.isnull().sum()`).
    *   Visualizations (using Seaborn and Matplotlib) to understand relationships between features and survival (e.g., survival counts, survival by sex/class, age/fare distributions, correlation heatmap).
4.  **Data Preprocessing:**
    *   **Handling Missing Values:**
        *   `Age`: Imputed with the median.
        *   `Embarked`: Imputed with the mode.
        *   `Cabin`: Dropped due to a high percentage of missing values.
    *   **Feature Engineering & Selection:**
        *   Irrelevant columns (`PassengerId`, `Name`, `Ticket`) dropped.
        *   Categorical features (`Sex`, `Embarked`) converted to numerical representations (mapping and one-hot encoding).
    *   **Data Splitting:** Data is separated into features (X) and the target variable (y).
    *   **Feature Scaling:** Numerical features (`Age`, `Fare`, `SibSp`, `Parch`) are scaled using Standardization (mean=0, std=1), as SVMs are sensitive to feature scales. This is done separately for the "from scratch" and `sklearn` sections.
5.  **SVM Implementation from Scratch (Section 5-7):**
    *   A Python `SVM` class is defined, implementing:
        *   Initialization of weights and bias.
        *   Hinge loss calculation with L2 regularization.
        *   Gradient descent algorithm for model training (`fit` method).
        *   Prediction function (`predict` method).
        *   Decision function to get raw scores.
    *   The custom SVM is trained on the manually scaled training data.
    *   Evaluation includes:
        *   Plotting the training loss curve.
        *   Calculating accuracy on training and test sets.
        *   Generating and plotting a confusion matrix.
        *   Calculating precision, recall, and F1-score.
        *   Manually calculating and plotting the ROC curve and AUC.
6.  **SVM using Scikit-learn (Section 8):**
    *   Relevant `sklearn` modules are imported (`train_test_split`, `StandardScaler`, `SVC`, metrics).
    *   Data is split using `train_test_split` (stratified).
    *   Features are scaled using `StandardScaler`.
    *   An `SVC` model (with a linear kernel for comparison) is instantiated and trained.
    *   Evaluation using `sklearn.metrics`:
        *   Accuracy score.
        *   Confusion matrix and plot.
        *   Classification report (precision, recall, F1-score).
        *   ROC curve and AUC calculation and plot.

## Dataset

The project utilizes the [Titanic dataset](https://www.kaggle.com/c/titanic/data), available in this repository as `Titanic-Dataset.csv`.

## Requirements

The necessary Python libraries are listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
```

## How to Run

1.  Ensure you have Python and pip installed.
2.  Clone this repository or download the project files.
3.  Navigate to the project directory in your terminal.
4.  Install the required libraries: `pip install -r requirements.txt`
5.  Launch Jupyter Notebook or JupyterLab and open `SVM_Classification.ipynb`:

    ```bash
    jupyter notebook SVM_Classification.ipynb
    # or
    jupyter lab SVM_Classification.ipynb
    ```
6.  Run the cells in the notebook sequentially to execute the analysis and model training/evaluation.
