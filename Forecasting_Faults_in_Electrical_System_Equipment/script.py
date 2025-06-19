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

# Définir les colonnes attendues pour la validation du fichier CSV
# Les colonnes 'Sample Day' et 'Tested day' ont été retirées de cette liste
EXPECTED_COLUMNS = [
    'LOC', 'NAME', 'CODETX', 'MFG', 'SER', 'KV', 'MVA', 'Year Test',
    'O2', 'N2', 'CO2', 'CO', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'C3H6', 'C3H8', 'TCG', 'TEMP', 'WATER'
]

def process_csv_data(csv_string):
    """
    Traite une chaîne CSV, valide les colonnes, gère les colonnes supplémentaires
    et effectue l'analyse des données.
    Retourne un dictionnaire de données traitées ou un dictionnaire d'erreur.
    """
    try:
        df = pd.read_csv(StringIO(csv_string))

        # --- Validation et gestion des colonnes ---
        current_columns = set(df.columns)
        expected_columns_set = set(EXPECTED_COLUMNS)

        missing_columns = list(expected_columns_set - current_columns)
        extra_columns = list(current_columns - expected_columns_set)

        error_messages = []
        if missing_columns:
            error_messages.append(f"Colonnes manquantes : {', '.join(missing_columns)}")
        
        # Si des colonnes manquantes sont détectées, retourner une erreur
        if error_messages:
            return {"error": True, "message": "Erreur de format de fichier : " + " ; ".join(error_messages)}

        # Supprimer les colonnes supplémentaires du DataFrame
        if extra_columns:
            df = df.drop(columns=extra_columns, errors='ignore')
            # Optionnel : vous pourriez ajouter un avertissement ici si vous vouliez loguer la suppression
            # print(f"Colonnes supplémentaires supprimées : {', '.join(extra_columns)}")

        # S'assurer que 'Year Test' n'a pas de valeurs manquantes et est un entier
        df = df.dropna(subset=["Year Test"])
        df["Year Test"] = df["Year Test"].astype(int)

        # Liste des colonnes à vérifier pour toutes les zéros (ajustée à la nouvelle liste)
        colonnes_a_verifier_zeros = EXPECTED_COLUMNS # Utilisez directement la liste attendue après la suppression des extras
        
        # Filtrer les lignes où toutes les colonnes à vérifier sont 0.0
        # S'assurer que le DataFrame contient toutes les colonnes avant de filtrer
        cols_to_check_present = [col for col in colonnes_a_verifier_zeros if col in df.columns]
        if cols_to_check_present:
            df = df[~df[cols_to_check_present].eq(0.0).all(axis=1)]
        
        # Calculer 'year_num' à partir de 'Year Test'
        df["year_num"] = df["Year Test"] - df["Year Test"].min()

        # Calculer les ratios de gaz
        for ratio, num, denom in zip(["R1", "R2", "R3", "R4", "R5"], ["CH4", "C2H2", "C2H2", "C2H6", "C2H4"], ["H2", "C2H4", "CH4", "C2H2", "C2H6"]):
            # Utiliser .get pour gérer les colonnes qui pourraient être manquantes après la suppression des extras
            # Bien que la validation initiale devrait garantir leur présence, c'est une bonne pratique
            if num in df.columns and denom in df.columns:
                df[ratio] = df[num] / df[denom].replace(0, np.nan)
            else:
                df[ratio] = np.nan # Assigner NaN si une colonne est manquante
        
        if "CH4" in df.columns and "C2H2" in df.columns and "C2H4" in df.columns:
            df["Duval_Total"] = df[["CH4", "C2H2", "C2H4"]].sum(axis=1).replace(0, np.nan)
        else:
            df["Duval_Total"] = np.nan

        # Fonctions de classification des défaillances
        def classify_IRM(row):
            # Vérifier la présence des colonnes avant d'y accéder
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

        # Appliquer les fonctions de classification
        for name, func in zip(["IRM", "DRM", "RRM", "DTM", "DPM"], [classify_IRM, classify_DRM, classify_RRM, classify_DTM, classify_DPM]):
            df[f"{name}_fault"] = df.apply(func, axis=1)

        # Calculer les pourcentages de gaz et les coordonnées Cx, Cy
        gases = ["CH4", "C2H2", "C2H4", "H2", "C2H6"]
        
        # S'assurer que toutes les colonnes de gaz sont présentes avant de faire la somme
        gases_present = [g for g in gases if g in df.columns]
        if gases_present:
            df["Total_DP"] = df[gases_present].sum(axis=1)
            df = df[df["Total_DP"] > 0] # Filtrer les lignes où Total_DP est zéro
            for g in gases_present:
                df[f"%{g}"] = 100 * df[g] / df["Total_DP"]
            # Assigner 0 aux pourcentages des gaz non présents pour éviter KeyError
            for g in [g for g in gases if g not in gases_present]:
                df[f"%{g}"] = 0.0
        else:
            df["Total_DP"] = 0.0 # Pas de gaz, pas de total
            for g in gases:
                df[f"%{g}"] = 0.0

        # Calculer Cx et Cy uniquement si les colonnes nécessaires existent
        if all(f"%{g}" in df.columns for g in ["CH4", "C2H6", "C2H2", "C2H4"]):
            df["Cx"] = (df["%CH4"] + 0.5 * df["%C2H6"] - 0.5 * df["%C2H2"] - df["%C2H4"]) / 100
            df["Cy"] = (0.866 * df["%C2H6"] + 0.866 * df["%C2H2"]) / 100
        else:
            df["Cx"] = np.nan
            df["Cy"] = np.nan


        # Encoder les colonnes de méthode de défaillance
        method_columns = ["IRM_fault", "DRM_fault", "RRM_fault", "DTM_fault", "DPM_fault"]
        for col in method_columns:
            if col in df.columns:
                df[f"{col}_enc"] = LabelEncoder().fit_transform(df[col])
            else:
                df[f"{col}_enc"] = -1 # ou une autre valeur par défaut pour indiquer l'absence

        # Déterminer la "vraie" défaillance basée sur le mode des méthodes
        method_columns_present = [col for col in method_columns if col in df.columns]
        if method_columns_present:
            df["true_fault"] = df[method_columns_present].mode(axis=1)[0]
        else:
            df["true_fault"] = "Uncertain" # Ou une valeur par défaut si aucune colonne de méthode n'est présente

        le_target = LabelEncoder()
        df["true_fault_index"] = le_target.fit_transform(df["true_fault"])
        fault_labels = {i: c for i, c in enumerate(le_target.classes_)}

        # Ajouter des caractéristiques temporelles
        df["year_sin"] = np.sin(2 * np.pi * df["Year Test"] / 12)
        df["year_cos"] = np.cos(2 * np.pi * df["Year Test"] / 12)

        # Préparer les données pour l'entraînement du modèle temporel
        X_temporal = df[["year_num", "year_sin", "year_cos"]]
        y_temporal = df["true_fault_index"].values

        # Générer des données futures pour la prédiction
        future_years = np.arange(df["Year Test"].min(), df["Year Test"].max() + 10) # Prédiction jusqu'à 2030
        future_year_num = future_years - df["Year Test"].min()
        future_year_sin = np.sin(2 * np.pi * future_years / 12)
        future_year_cos = np.cos(2 * np.pi * future_years / 12)
        X_future_temporal = pd.DataFrame({"year_num": future_year_num, "year_sin": future_year_sin, "year_cos": future_year_cos})

        # Calculer les proportions réelles de défaillances par année
        real_proportions_by_year = df.groupby("Year Test")["true_fault_index"].value_counts(normalize=True).unstack(fill_value=0)
        real_proportions_by_year.columns = [fault_labels[idx] for idx in real_proportions_by_year.columns]

        # Définir les modèles à évaluer
        models_to_evaluate = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        # Définir les grilles de paramètres pour GridSearchCV
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

        # Normaliser les caractéristiques temporelles
        scaler_temporal = StandardScaler()
        X_temporal_scaled = scaler_temporal.fit_transform(X_temporal)
        X_temporal_scaled_df = pd.DataFrame(X_temporal_scaled, columns=X_temporal.columns, index=X_temporal.index)

        # Configurer TimeSeriesSplit pour la validation croisée
        tscv = TimeSeriesSplit(n_splits=5)
        all_model_predictions = {}
        results_temporal = {}

        # Entraîner et évaluer les modèles
        for name, model_instance in models_to_evaluate.items():
            model = model_instance
            if name in param_grids:
                grid_search = GridSearchCV(model, param_grids[name], cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
                grid_search.fit(X_temporal_scaled_df, y_temporal)
                model = grid_search.best_estimator_
            else:
                model.fit(X_temporal_scaled_df, y_temporal)

            # Faire des prédictions de probabilité pour les années futures
            if hasattr(model, "predict_proba"):
                X_future_scaled = scaler_temporal.transform(X_future_temporal)
                proba = model.predict_proba(X_future_scaled)
                df_pred = pd.DataFrame(0.0, index=future_years, columns=[fault_labels[i] for i in range(len(fault_labels))])
                for i, class_idx in enumerate(model.classes_):
                    if class_idx in fault_labels: # S'assurer que class_idx existe dans fault_labels
                        df_pred[fault_labels[class_idx]] = proba[:, i]
                all_model_predictions[name] = df_pred

            # Calculer les métriques de performance
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

        # Créer un DataFrame des métriques et identifier le meilleur modèle
        metrics_df_temporal = pd.DataFrame(results_temporal).T.round(3)
        metrics_df_temporal.sort_values("Accuracy", ascending=False, inplace=True)
        best_temporal_model_name = metrics_df_temporal.index[0] if not metrics_df_temporal.empty else list(models_to_evaluate.keys())[0]

        # Retourner toutes les données traitées et les résultats
        return {
            "error": False, # Indique qu'il n'y a pas d'erreur
            "df": df,
            "real_proportions_by_year": real_proportions_by_year,
            "all_model_predictions": all_model_predictions,
            "metrics_df_temporal": metrics_df_temporal,
            "fault_labels": fault_labels,
            "best_temporal_model_name": best_temporal_model_name
        }
    except Exception as e:
        # Gérer toute autre erreur inattendue lors du traitement des données
        print(f"Erreur lors du traitement des données : {e}")
        return {"error": True, "message": f"Erreur lors du traitement des données : {str(e)}"}

