import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

def process_csv_data(csv_string):
    df = pd.read_csv(StringIO(csv_string))
    # Drop the 'NB' column as specified, and previous unnamed columns if they exist
    df = df.drop(columns=["NB"], errors='ignore')
    df = df.dropna(subset=["Year Test"])
    df["Year Test"] = df["Year Test"].astype(int)

    # Updated list of columns to check for all zeros, including new ones
    colonnes_a_verifier = [
        'LOC', 'NAME', 'CODETX', 'MFG', 'SER', 'KV', 'MVA', 'Year Test', 'Sample Day', 'Tested day',
        'O2', 'N2', 'CO2', 'CO', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'C3H6', 'C3H8', 'TCG', 'TEMP', 'WATER'
    ]
    df = df[~df[colonnes_a_verifier].eq(0.0).all(axis=1)]
    df["year_num"] = df["Year Test"] - df["Year Test"].min()

    for ratio, num, denom in zip(["R1", "R2", "R3", "R4", "R5"], ["CH4", "C2H2", "C2H2", "C2H6", "C2H4"], ["H2", "C2H4", "CH4", "C2H2", "C2H6"]):
        df[ratio] = df[num] / df[denom].replace(0, np.nan)
    df["Duval_Total"] = df[["CH4", "C2H2", "C2H4"]].sum(axis=1).replace(0, np.nan)

    def classify_IRM(row):
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
        if pd.isna(row["R1"]) or pd.isna(row["R2"]) or pd.isna(row["R3"]) or pd.isna(row["R4"]): return "Uncertain"
        if not (row["H2"] > 100 and (row["CH4"] > 120 or row["C2H2"] > 1 or row["C2H4"] > 50)): return "Uncertain"
        R1, R2, R3, R4 = row["R1"], row["R2"], row["R3"], row["R4"]
        if R1 > 1 and R2 < 0.75 and R3 < 0.3 and R4 > 0.4: return "F1"
        elif R1 < 0.1 and R3 < 0.3 and R4 > 0.4: return "PD"
        elif 0.1 <= R1 <= 1 and R2 > 0.75 and R3 < 0.3 and R4 < 0.4: return "D2"
        else: return "Uncertain"

    def classify_RRM(row):
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

    for name, func in zip(["IRM", "DRM", "RRM", "DTM", "DPM"], [classify_IRM, classify_DRM, classify_RRM, classify_DTM, classify_DPM]):
        df[f"{name}_fault"] = df.apply(func, axis=1)

    gases = ["CH4", "C2H2", "C2H4", "H2", "C2H6"]
    df["Total_DP"] = df[gases].sum(axis=1)
    df = df[df["Total_DP"] > 0]
    for g in gases:
        df[f"%{g}"] = 100 * df[g] / df["Total_DP"]
    df["Cx"] = (df["%CH4"] + 0.5 * df["%C2H6"] - 0.5 * df["%C2H2"] - df["%C2H4"]) / 100
    df["Cy"] = (0.866 * df["%C2H6"] + 0.866 * df["%C2H2"]) / 100

    method_columns = ["IRM_fault", "DRM_fault", "RRM_fault", "DTM_fault", "DPM_fault"]
    for col in method_columns:
        df[f"{col}_enc"] = LabelEncoder().fit_transform(df[col])

    df["true_fault"] = df[method_columns].mode(axis=1)[0]
    le_target = LabelEncoder()
    df["true_fault_index"] = le_target.fit_transform(df["true_fault"])
    fault_labels = {i: c for i, c in enumerate(le_target.classes_)}

    df["year_sin"] = np.sin(2 * np.pi * df["Year Test"] / 12)
    df["year_cos"] = np.cos(2 * np.pi * df["Year Test"] / 12)

    X_temporal = df[["year_num", "year_sin", "year_cos"]]
    y_temporal = df["true_fault_index"].values

    future_years = np.arange(df["Year Test"].min(), 2031)
    future_year_num = future_years - df["Year Test"].min()
    future_year_sin = np.sin(2 * np.pi * future_years / 12)
    future_year_cos = np.cos(2 * np.pi * future_years / 12)
    X_future_temporal = pd.DataFrame({"year_num": future_year_num, "year_sin": future_year_sin, "year_cos": future_year_cos})

    real_proportions_by_year = df.groupby("Year Test")["true_fault_index"].value_counts(normalize=True).unstack(fill_value=0)
    real_proportions_by_year.columns = [fault_labels[idx] for idx in real_proportions_by_year.columns]

    models_to_evaluate = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        # CORRECTION ICI : 'random_ado' a été changé en 'random_state'
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

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

    scaler_temporal = StandardScaler()
    X_temporal_scaled = scaler_temporal.fit_transform(X_temporal)
    X_temporal_scaled_df = pd.DataFrame(X_temporal_scaled, columns=X_temporal.columns, index=X_temporal.index)

    tscv = TimeSeriesSplit(n_splits=5)
    all_model_predictions = {}
    results_temporal = {}

    for name, model_instance in models_to_evaluate.items():
        model = model_instance
        if name in param_grids:
            grid_search = GridSearchCV(model, param_grids[name], cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_temporal_scaled_df, y_temporal)
            model = grid_search.best_estimator_
        else:
            model.fit(X_temporal_scaled_df, y_temporal)

        if hasattr(model, "predict_proba"):
            X_future_scaled = scaler_temporal.transform(X_future_temporal)
            proba = model.predict_proba(X_future_scaled)
            df_pred = pd.DataFrame(0.0, index=future_years, columns=[fault_labels[i] for i in range(len(fault_labels))])
            for i, class_idx in enumerate(model.classes_):
                if class_idx in fault_labels: # Ensure class_idx exists in fault_labels
                    df_pred[fault_labels[class_idx]] = proba[:, i]
            all_model_predictions[name] = df_pred

        y_pred = model.predict(X_temporal_scaled_df)
        acc = accuracy_score(y_temporal, y_pred)
        prec = precision_score(y_temporal, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_temporal, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_temporal, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_temporal, y_pred)
        specificity = []
        # Calculate specificity for each class
        for i in range(len(fault_labels)):
            # Check if the class index exists in the confusion matrix dimensions
            if i < cm.shape[0] and i < cm.shape[1]:
                TP = cm[i, i]
                # Sum across the row for False Negatives, excluding the TP
                FN = cm[i, :].sum() - TP
                # Sum across the column for False Positives, excluding the TP
                FP = cm[:, i].sum() - TP
                # Total elements minus (TP + FN + FP)
                TN = cm.sum() - (TP + FN + FP)
                specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
            else:
                specificity.append(0) # If class not present in CM, specificity is 0 or undefined

        results_temporal[name] = {
            "Accuracy": acc, "Precision": prec, "Recall": rec,
            "Specificity": np.mean(specificity), "F1": f1
        }

    metrics_df_temporal = pd.DataFrame(results_temporal).T.round(3)
    metrics_df_temporal.sort_values("Accuracy", ascending=False, inplace=True)
    best_temporal_model_name = metrics_df_temporal.index[0] if not metrics_df_temporal.empty else list(models_to_evaluate.keys())[0]

    return {
        "df": df,
        "real_proportions_by_year": real_proportions_by_year,
        "all_model_predictions": all_model_predictions,
        "metrics_df_temporal": metrics_df_temporal,
        "fault_labels": fault_labels,
        "best_temporal_model_name": best_temporal_model_name
    }
