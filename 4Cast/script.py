import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from io import StringIO
# warnings.filterwarnings('ignore') # Removed as requested

# Define expected columns for CSV file validation for TGA
EXPECTED_COLUMNS_TGA = [
    'LOC', 'NAME', 'CODETX', 'MFG', 'SER', 'KV', 'MVA', 'Year Test',
    'O2', 'N2', 'CO2', 'CO', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'C3H6', 'C3H8', 'TCG', 'TEMP', 'WATER'
]

def process_csv_data_temporal(csv_string):
    """
    Processes a CSV string for TGA analysis, validates columns, handles extra columns,
    and performs DGA data analysis and temporal modeling.
    Returns a dictionary of processed data or an error dictionary.
    """
    try:
        df = pd.read_csv(StringIO(csv_string))

        # --- Column Validation and Handling ---
        current_columns = set(df.columns)
        expected_columns_set = set(EXPECTED_COLUMNS_TGA)

        missing_columns = list(expected_columns_set - current_columns)
        
        error_messages = []
        if missing_columns:
            error_messages.append(f"Missing columns: {', '.join(missing_columns)}")
        
        # If missing columns are detected, return an error
        if error_messages:
            return {"error": True, "message": "File format error: " + " ; ".join(error_messages)}

        # Remove extra columns from the DataFrame
        extra_columns = list(current_columns - expected_columns_set)
        if extra_columns:
            df = df.drop(columns=extra_columns, errors='ignore')

        # Ensure 'Year Test' has no missing values and is an integer
        df = df.dropna(subset=["Year Test"])
        df["Year Test"] = df["Year Test"].astype(int)

        # Filter out rows where all checked columns are 0.0
        cols_to_check_present = [col for col in EXPECTED_COLUMNS_TGA if col in df.columns]
        if cols_to_check_present:
            df = df[~df[cols_to_check_present].eq(0.0).all(axis=1)]
        
        # Calculate 'year_num' from 'Year Test'
        df["year_num"] = df["Year Test"] - df["Year Test"].min()

        # Calculate gas ratios
        for ratio, num, denom in zip(["R1", "R2", "R3", "R4", "R5"], ["CH4", "C2H2", "C2H2", "C2H6", "C2H4"], ["H2", "C2H4", "CH4", "C2H2", "C2H6"]):
            if num in df.columns and denom in df.columns:
                df[ratio] = df[num] / df[denom].replace(0, np.nan)
            else:
                df[ratio] = np.nan
        
        if "CH4" in df.columns and "C2H2" in df.columns and "C2H4" in df.columns:
            df["Duval_Total"] = df[["CH4", "C2H2", "C2H4"]].sum(axis=1).replace(0, np.nan)
        else:
            df["Duval_Total"] = np.nan

        # Failure classification functions
        def classify_IRM(row):
            if "R1" not in row or "R2" not in row or "R5" not in row: return "Uncertain"
            R1, R2, R5 = row["R1"], row["R2"], row["R5"]
            if pd.isna(R1) or pd.isna(R2) or pd.isna(R5): return "Uncertain"
            if R1 < 0.1 and R5 < 0.2: return "PD"
            elif R1 < 0.1 and R2 > 1 and R5 > 1: return "PD"
            elif 0.1 <= R1 <= 1 and 0.6 <= R2 <= 2.5 and R5 > 2: return "D1"
            elif R5 < 1: return "F1"
            elif R1 > 1 and R2 < 0.1 and 1 <= R5 <= 4: return "T2"
            elif R1 > 1 and R2 < 0.2 and R5 > 4: return "T3"
            else: return "Uncertain"

        def classify_DRM(row):
            if not all(col in row for col in ["R1", "R2", "R3", "R4", "H2", "CH4", "C2H2", "C2H4"]): return "Uncertain"
            if pd.isna(row["R1"]) or pd.isna(row["R2"]) or pd.isna(row["R3"]) or pd.isna(row["R4"]): return "Uncertain"
            if not (row["H2"] > 100 and (row["CH4"] > 120 or row["C2H2"] > 1 or row["C2H4"] > 50)): return "Uncertain"
            R1, R2, R3, R4 = row["R1"], row["R2"], row["R3"], row["R4"]
            if R1 > 1 and R2 < 0.75 and R3 < 0.3 and R4 > 0.4: return "F1"
            elif R1 < 0.1 and R3 < 0.3 and R4 > 0.4: return "PD"
            elif 0.1 <= R1 <= 1 and R2 > 0.75 and R3 < 0.3 and R4 < 0.4: return "D2"
            else: return "Uncertain"

        def classify_RRM(row):
            if not all(col in row for col in ["R1", "R2", "R5", "CH4", "H2", "C2H2", "C2H4", "C2H6"]): return "Uncertain"
            if pd.isna(row["R1"]) or pd.isna(row["R2"]) or pd.isna(row["R5"]): return "Uncertain"
            if row[["CH4", "H2", "C2H2", "C2H4", "C2H6"]].min() <= 0: return "Uncertain"
            R1, R2, R5 = row["R1"], row["R2"], row["R5"]
            if 0.1 <= R1 <= 1 and R2 < 0.1 and R5 < 1: return "F1"
            elif R1 < 0.1 and R2 < 0.1 and R5 < 1: return "PD"
            elif 0.1 <= R1 <= 1 and 0.1 <= R2 <= 3 and R5 > 3: return "D2"
            elif 0.1 <= R1 <= 1 and R2 < 0.1 and 1 <= R5 <= 3: return "T1"
            elif R1 > 1 and R2 < 0.1 and 1 <= R5 <= 3: return "T2"
            elif R1 > 1 and R2 < 0.1 and R5 > 3: return "T3"
            else: return "Uncertain"

        def classify_DTM(row):
            if not all(col in row for col in ["CH4", "C2H2", "C2H4"]): return "Uncertain"
            ch4, c2h2, c2h4 = row["CH4"], row["C2H2"], row["C2H4"]
            total = ch4 + c2h2 + c2h4
            if total == 0: return "Uncertain"
            pct_ch4 = 100 * ch4 / total
            pct_c2h2 = 100 * c2h2 / total
            pct_c2h4 = 100 * c2h4 / total
            if pct_c2h2 > 23 and pct_c2h4 < 40: return "PD"
            elif pct_c2h4 > 40 and pct_c2h2 < 23: return "T2"
            elif 23 <= pct_c2h2 <= 50 and 20 <= pct_c2h4 <= 50: return "D1"
            elif pct_c2h4 < 20 and pct_c2h2 < 23: return "T1"
            else: return "Uncertain"

        def classify_DPM(row):
            gases = ["CH4", "C2H2", "C2H4", "H2", "C2H6"]
            if not all(g in row for g in gases): return "Uncertain"
            total = row[gases].sum()
            if total == 0: return "Uncertain"
            pct_ch4 = 100 * row["CH4"] / total
            pct_c2h2 = 100 * row["C2H2"] / total
            pct_c2h4 = 100 * row["C2H4"] / total
            pct_h2 = 100 * row["H2"] / total
            pct_c2h6 = 100 * row["C2H6"] / total
            if pct_c2h2 > 20 and pct_h2 > 15: return "PD"
            elif pct_c2h4 > 40 and pct_c2h6 < 10: return "T1"
            elif pct_c2h6 > 10 and pct_ch4 > 15: return "T3"
            else: return "Uncertain"

        # Apply classification functions
        for name, func in zip(["IRM", "DRM", "RRM", "DTM", "DPM"], [classify_IRM, classify_DRM, classify_RRM, classify_DTM, classify_DPM]):
            df[f"{name}_fault"] = df.apply(func, axis=1)

        # Calculate gas percentages and Cx, Cy coordinates
        gases = ["CH4", "C2H2", "C2H4", "H2", "C2H6"]
        
        gases_present = [g for g in gases if g in df.columns]
        if gases_present:
            df["Total_DP"] = df[gases_present].sum(axis=1)
            df = df[df["Total_DP"] > 0]
            for g in gases_present:
                df[f"%{g}"] = 100 * df[g] / df["Total_DP"]
            for g in [g for g in gases if g not in gases_present]:
                df[f"%{g}"] = 0.0
        else:
            df["Total_DP"] = 0.0
            for g in gases:
                df[f"%{g}"] = 0.0

        if all(f"%{g}" in df.columns for g in ["CH4", "C2H6", "C2H2", "C2H4"]):
            df["Cx"] = (df["%CH4"] + 0.5 * df["%C2H6"] - 0.5 * df["%C2H2"] - df["%C2H4"]) / 100
            df["Cy"] = (0.866 * df["%C2H6"] + 0.866 * df["%C2H2"]) / 100
        else:
            df["Cx"] = np.nan
            df["Cy"] = np.nan


        # Encode fault method columns
        method_columns = ["IRM_fault", "DRM_fault", "RRM_fault", "DTM_fault", "DPM_fault"]
        for col in method_columns:
            if col in df.columns:
                df[f"{col}_enc"] = LabelEncoder().fit_transform(df[col])
            else:
                df[f"{col}_enc"] = -1

        # Determine the "true" fault based on the mode of the methods
        method_columns_present = [col for col in method_columns if col in df.columns]
        if method_columns_present:
            df["true_fault"] = df[method_columns_present].mode(axis=1)[0]
        else:
            df["true_fault"] = "Uncertain"

        le_target = LabelEncoder()
        df["true_fault_index"] = le_target.fit_transform(df["true_fault"])
        fault_labels = {i: c for i, c in enumerate(le_target.classes_)}

        # Add temporal features
        df["year_sin"] = np.sin(2 * np.pi * df["Year Test"] / 12)
        df["year_cos"] = np.cos(2 * np.pi * df["Year Test"] / 12)

        # Prepare data for temporal model training
        X_temporal = df[["year_num", "year_sin", "year_cos"]]
        y_temporal = df["true_fault_index"].values

        # Generate future data for prediction
        future_years = np.arange(df["Year Test"].min(), df["Year Test"].max() + 10)
        future_year_num = future_years - df["Year Test"].min()
        future_year_sin = np.sin(2 * np.pi * future_years / 12)
        future_year_cos = np.cos(2 * np.pi * future_years / 12)
        X_future_temporal = pd.DataFrame({"year_num": future_year_num, "year_sin": future_year_sin, "year_cos": future_year_cos})

        # Calculate actual fault proportions by year
        real_proportions_by_year = df.groupby("Year Test")["true_fault_index"].value_counts(normalize=True).unstack(fill_value=0)
        real_proportions_by_year.columns = [fault_labels[idx] for idx in real_proportions_by_year.columns]

        # Define models to evaluate
        models_to_evaluate = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        # Define parameter grids for GridSearchCV
        param_grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            "KNN": {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            "SVM": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            "Logistic Regression": {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear'],
                'penalty': ['l2']
            },
            "Decision Tree": {
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        }

        # Scale temporal features
        scaler_temporal = StandardScaler()
        X_temporal_scaled = scaler_temporal.fit_transform(X_temporal)
        X_temporal_scaled_df = pd.DataFrame(X_temporal_scaled, columns=X_temporal.columns, index=X_temporal.index)

        # Configure TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        all_model_predictions = {}
        results_temporal = {}

        # Train and evaluate models
        for name, model_instance in models_to_evaluate.items():
            model = model_instance
            if name in param_grids:
                grid_search = GridSearchCV(model, param_grids[name], cv=tscv, scoring='accuracy', n_jobs=-1, verbose=0) # Set verbose to 0
                grid_search.fit(X_temporal_scaled_df, y_temporal)
                model = grid_search.best_estimator_
            else:
                model.fit(X_temporal_scaled_df, y_temporal)

            # Make probability predictions for future years
            if hasattr(model, "predict_proba"):
                X_future_scaled = scaler_temporal.transform(X_future_temporal)
                proba = model.predict_proba(X_future_scaled)
                df_pred = pd.DataFrame(0.0, index=future_years, columns=[fault_labels[i] for i in range(len(fault_labels))])
                for i, class_idx in enumerate(model.classes_):
                    if class_idx in fault_labels:
                        df_pred[fault_labels[class_idx]] = proba[:, i]
                all_model_predictions[name] = df_pred

            # Calculate performance metrics
            y_pred = model.predict(X_temporal_scaled_df)
            acc = accuracy_score(y_temporal, y_pred)
            prec = precision_score(y_temporal, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_temporal, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_temporal, y_pred, average='macro', zero_division=0)
            cm = confusion_matrix(y_temporal, y_pred)
            specificity = []
            for i in range(len(fault_labels)):
                if i < cm.shape[0] and i < cm.shape[1]:
                    TP = cm[i, i]
                    FN = cm[i, :].sum() - TP
                    FP = cm[:, i].sum() - TP
                    TN = cm.sum() - (TP + FN + FP)
                    specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
                else:
                    specificity.append(0)

            results_temporal[name] = {
                "Accuracy": acc, "Precision": prec, "Recall": rec,
                "Specificity": np.mean(specificity), "F1": f1
            }

        # Create a DataFrame of metrics and identify the best model
        metrics_df_temporal = pd.DataFrame(results_temporal).T.round(3)
        metrics_df_temporal.sort_values("Accuracy", ascending=False, inplace=True)
        best_temporal_model_name = metrics_df_temporal.index[0] if not metrics_df_temporal.empty else list(models_to_evaluate.keys())[0]

        # Return all processed data and results
        return {
            "error": False,
            "df": df,
            "real_proportions_by_year": real_proportions_by_year,
            "all_model_predictions": all_model_predictions,
            "metrics_df_temporal": metrics_df_temporal,
            "fault_labels": fault_labels,
            "best_temporal_model_name": best_temporal_model_name
        }
    except Exception as e:
        print(f"Error during TGA data processing: {e}")
        return {"error": True, "message": f"Error during TGA data processing: {str(e)}"}
    

def process_custom_data(df_json_string, xaxis_col, yaxis_col):
    """
    Processes a custom CSV string for general temporal prediction.
    It trains regression models to predict yaxis_col based on xaxis_col.
    Returns a dictionary of processed data and prediction results.
    """
    try:
        df = pd.read_json(StringIO(df_json_string), orient='split')

        # Ensure selected columns are numeric and handle missing values
        df[xaxis_col] = pd.to_numeric(df[xaxis_col], errors='coerce')
        df[yaxis_col] = pd.to_numeric(df[yaxis_col], errors='coerce')
        df.dropna(subset=[xaxis_col, yaxis_col], inplace=True)

        if df.empty:
            return {"error": True, "message": "No valid data after cleaning for selected columns."}

        # Prepare data for modeling
        X = df[[xaxis_col]]
        y = df[yaxis_col]

        # Define regression models to evaluate
        models_to_evaluate = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
            "SVM": SVR()
        }

        # Define parameter grids for GridSearchCV for regression models
        param_grids = {
            "Random Forest": {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
            },
            "KNN": {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            "SVM": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            "Decision Tree": {
                'max_depth': [3, 5, 7],
            }
        }

        all_model_predictions = {}
        results_metrics = {}

        # Sort data by the x-axis column for time series split
        df_sorted = df.sort_values(by=xaxis_col).reset_index(drop=True)
        X_sorted = df_sorted[[xaxis_col]]
        y_sorted = df_sorted[yaxis_col]

        # Use TimeSeriesSplit for cross-validation if x-axis is temporal, else KFold
        # Assuming xaxis_col is often temporal, but if not, KFold might be better
        # For simplicity, we will use TimeSeriesSplit for all cases here given the user's focus on temporal data.
        # However, for truly non-temporal X, a standard KFold or train_test_split would be more appropriate.
        try:
            tscv = TimeSeriesSplit(n_splits=min(5, len(df_sorted) // 2)) # Ensure n_splits is not too large
        except ValueError: # Not enough data for 5 splits
            tscv = TimeSeriesSplit(n_splits=2) # Fallback to 2 splits or fewer

        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_sorted)

        # Generate future data for prediction (e.g., 10 future points beyond max X value)
        max_x = X_sorted[xaxis_col].max()
        # Create future X values, assuming a step similar to the average step in the data
        if len(X_sorted) > 1:
            avg_step = X_sorted[xaxis_col].diff().mean()
        else:
            avg_step = 1 # Default step if only one data point

        future_x_values = np.arange(max_x + avg_step, max_x + (10 * avg_step) + 0.1, avg_step)
        X_future = pd.DataFrame(future_x_values, columns=[xaxis_col])
        X_future_scaled = scaler_X.transform(X_future)


        for name, model_instance in models_to_evaluate.items():
            model = model_instance
            if name in param_grids:
                grid_search = GridSearchCV(model, param_grids[name], cv=tscv, scoring='r2', n_jobs=-1, verbose=0) # Set verbose to 0
                grid_search.fit(X_scaled, y_sorted)
                model = grid_search.best_estimator_
            else:
                model.fit(X_scaled, y_sorted)

            # Make predictions for future values
            future_predictions = model.predict(X_future_scaled)
            df_pred = pd.DataFrame(future_predictions, index=future_x_values, columns=[yaxis_col])
            all_model_predictions[name] = df_pred

            # Calculate performance metrics on the training data
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y_sorted, y_pred)
            mae = mean_absolute_error(y_sorted, y_pred)
            mse = mean_squared_error(y_sorted, y_pred)
            rmse = np.sqrt(mse)

            results_metrics[name] = {
                "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse
            }

        metrics_df = pd.DataFrame(results_metrics).T.round(3)
        metrics_df.sort_values("R2", ascending=False, inplace=True)

        return {
            "error": False,
            "actual_data": df.to_json(date_format='iso', orient='split'),
            "all_model_predictions": {k: v.to_json(date_format='iso', orient='split') for k, v in all_model_predictions.items()},
            "metrics_df": metrics_df.to_json(date_format='iso', orient='split'),
            "best_model_name": metrics_df.index[0] if not metrics_df.empty else list(models_to_evaluate.keys())[0]
        }

    except Exception as e:
        print(f"Error during custom data processing: {e}")
        return {"error": True, "message": f"Error during custom data processing: {str(e)}"}
