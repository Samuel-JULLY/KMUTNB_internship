# 4Cast: Data-driven Power Transformer Health Assessment

## 1. What is 4Cast?
4Cast is a comprehensive diagnostic framework and interactive application designed to enhance the health assessment and fault diagnosis of power transformers. It synergizes traditional Dissolved Gas Analysis (DGA) techniques with advanced machine learning models to provide a more robust, accurate, and consistent diagnostic solution compared to conventional interpretation methods.

The application offers two main modes:

- **TGA (Transformer Gas Analysis) Mode:** Specifically tailored for DGA data, providing in-depth analysis of transformer health, fault detection, and temporal predictions.

- **Other Mode:** A flexible mode allowing users to upload any CSV file, select X and Y axes, and perform custom temporal analysis and predictions using various regression models.


## 2. Objective and Methodology
The primary objective of 4Cast is to overcome the limitations of conventional DGA interpretation methods, which often suffer from inconsistencies due to static thresholds and reliance on expert judgment.

The methodology involves:

- **Data Preprocessing and Feature Engineering:** Rigorous validation, cleaning, and transformation of raw CSV data. This includes handling missing values, calculating essential gas ratios (R1, R2, R3, R4, R5), computing Duval Triangle parameters (Duval_Total, percentages, Cx/Cy coordinates), and creating temporal features (`year_num`, `year_sin`, `year_cos`). Numeric features are scaled using `StandardScaler`, and categorical fault types are encoded with `LabelEncoder`.

- **Baseline Diagnostic Methods:** Integration of five conventional DGA interpretation methods for comparative insights:

    - IEC Ratio Method (IRM)

    - Duval Pentagon Method (DPM)

    - Rogers Ratio Method (RRM)

    - Doernenburg Ratio Method (DRM)

    - Duval Triangle Method (DTM)

- **Machine Learning Model Development:** Implementation of a diverse set of supervised machine learning models for both classification (in TGA mode) and regression (in Other mode) tasks.

- **Temporal Modeling:** Emphasis on predicting future fault trends or target values over time.

- **Interactive Visualization:** Providing an intuitive interface for users to upload datasets, visualize actual vs. predicted temporal trends, and evaluate model performance.


## 3. Machine Learning Models Used
4Cast leverages a variety of machine learning algorithms, carefully selected for their effectiveness in diagnostic applications and their ability to capture complex relationships within data.

### **For Classification Tasks (primarily in TGA Mode):**

- **Random Forest Classifier:** An ensemble method that builds multiple decision trees to improve accuracy and control overfitting.

- **K-Nearest Neighbors Classifier (KNeighbors Classifier):** A non-parametric method classifying data points based on the majority class of their 'k' nearest neighbors.

- **Support Vector Classifier (SVC):** A powerful model that finds an optimal hyperplane for class separation.

- **Logistic Regression:** A linear model for binary and multi-class classification, estimating class probabilities.

- **Decision Tree Classifier:** A tree-like model that makes decisions by splitting data into subsets.

- **Gaussian Naive Bayes:** A probabilistic classifier based on Bayes' theorem.


### **For Regression Tasks (primarily in Other Mode):**

- **Random Forest Regressor:** Adapted from the classifier for regression problems by averaging predictions of individual trees.

- **K-Nearest Neighbors Regressor (KNeighbors Regressor):** Predicts values by averaging the values of 'k' nearest neighbors.

- **Support Vector Regressor (SVR):** Applies SVM principles to regression, finding a function that fits data within a specified margin.

- **Linear Regression:** A fundamental statistical model fitting a linear equation to observed data.

- **Decision Tree Regressor:** A tree-like model for regression, where the predicted value is the average of target values in a leaf node.

Model training incorporates `TimeSeriesSplit` for robust cross-validation on temporal data and `GridSearchCV` for hyperparameter tuning to optimize performance.


## 4. Requirements for Launching the Application
To run the 4Cast application, you will need:

**Python 3.11.9:** The application is built with Python.

**Required Python Libraries:** All necessary libraries are listed in `app.py` and `script.py`. These include:

- `dash`
- `plotly`
- `pandas`
- `numpy`
- `scikit-learn`
- `json`
- `base64`
- `io`
- `webbrowser`
- `threading`
- `time`
- `os`
- `signal`
- `flask`
- `socket`
- `sys`

**CSV Data:** For TGA analysis, a CSV file with specific columns (`LOC`, `NAME`, `CODETX`, `MFG`, `SER`, `KV`, `MVA`, `Year Test`, `O2`, `N2`, `CO2`, `CO`, `H2`, `CH4`, `C2H2`, `C2H4`, `C2H6`, `C3H6`, `C3H8`, `TCG`, `TEMP`, `WATER`) is required. For "Other" mode, any structured CSV file can be used.

**Internet Connection:** Required for loading external stylesheets (Font Awesome) and for the AI-powered chatbot (if enabled and accessible via Gradio link).


## 5. How to Launch the Application
The application is designed to run locally and open in your web browser. 

Run the application:
Double-click the `4Cast_Application.bat` file.
This will:

- Activate the Python virtual environment.
- Execute `app.py`.
- The application will automatically open in your default web browser at `http://127.0.0.1:8050/`.
- A console window will remain open, displaying application logs. Do not close this window until you wish to shut down the application.


## 6. Key Features
- Interactive Dashboard: User-friendly interface for data upload, analysis, and visualization.
- Dynamic Filtering: Filter data by model, CODETX, MFG, fault type, and year range.
- Temporal Trend Visualization: Graphs display actual data points against selected model predictions over time, showing fault proportions or target values.
- Performance Metrics: Dynamic display of model performance metrics (Accuracy, Precision, Recall, F1-Score for classification; R², MAE, MSE, RMSE for regression).
- Detailed Prediction Tables: Tables showing future fault probabilities or predicted values, broken down by transformer ID and model.
- Single Instance Lock: Ensures only one instance of the application runs at a time, preventing port conflicts.
- AI-Powered Chatbot: An integrated chatbot (powered by a specialized LLM like Phi-3-mini-4k-instruct) to answer questions strictly related to electrical faults, providing comprehensive support and troubleshooting guidance. It can also perform online research for information beyond its training data cutoff.

## 7. Authors

Samuel Jully & Valentin Obert

CESI Engineering School, France, Reims
