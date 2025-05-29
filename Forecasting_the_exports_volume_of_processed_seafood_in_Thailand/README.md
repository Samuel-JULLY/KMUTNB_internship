
# Forecasting the exports volume of processed seafood in Thailand

## Project Overview

This project aims to forecast monthly export volumes of processed seafood products from Thailand through December 2025. It uses historical time series data (2007–2024) and compares several regression models, with and without hyperparameter optimization, to evaluate forecasting performance.

## Objectives

- Load, clean, and preprocess seafood export data

- Generate temporal features (trend and seasonality)

- Train and evaluate multiple regression models

- Optimize model performance using cross-validation and grid search

- Forecast monthly export volumes through 2025

- Visualize and compare model predictions


# Methodology

1. Data Preprocessing

    - Remove rows with zero values across all product columns

    - Convert Year and Month into datetime and numeric time index

    - Create seasonal features using sine and cosine of the month

2. Modeling

    - Use time series cross-validation (5-fold) to evaluate the model

    - Metrics: RMSE (Root Mean Squared Error) and R² Score

3. Hyperparameter Optimization

    - Perform GridSearchCV on SVR, Decision Tree, Random Forest, and KNN

    - Evaluate improvements post-tuning

4. Forecasting

    - Generate month-by-month predictions through December 2025

    - Forecast both total exports and category-specific volumes

## Models Used

- Linear Regression

- Support Vector Regression (SVR)

- Decision Tree Regressor

- Random Forest Regressor

- K-Nearest Neighbors Regressor (KNN)

## Visualizations

- Line plots: historical vs predicted values

- Scatter plots: actual vs predicted values per fold

- Summary tables: RMSE and R² (mean ± std) for each model

- Combined forecast plots (2007–2025) for all models

## Key Results

- Random Forest achieved the best balance of accuracy and stability after hyperparameter tuning.

- KNN produced competitive RMSE scores but could not extrapolate beyond 2024.

- Linear models (Linear Regression and SVR) failed to capture non-linear temporal patterns.

## Project Structure

.

├── DATA/

│   ├── Export 2007-2024.xlsx

│   └── Export 2007-2024.csv

├── notebook/

│   └── forecasting_project.ipynb

├── Projet.html

├── README.md

## Requirements

To run this project, you need Python ≥ 3.8 and the following libraries:

Required Python Packages : 

| Package        | Purpose                                                                                  |
| -------------- | ---------------------------------------------------------------------------------------- |
| `pandas`       | Data loading and manipulation                                                            |
| `numpy`        | Numerical operations and math functions                                                  |
| `matplotlib`   | Data visualization (line and scatter plots)                                              |
| `scikit-learn` | Machine learning models, evaluation metrics, cross-validation, and hyperparameter tuning |
| `openpyxl`     | Required by `pandas` to read `.xlsx` Excel files                                         |

You can install all dependencies using pip:

``` python
pip install pandas numpy matplotlib scikit-learn openpyxl
```

## Authors

Samuel JULLY & Valentin OBERT

CESI GRADUATE SCHOOL OF ENGINEERING


