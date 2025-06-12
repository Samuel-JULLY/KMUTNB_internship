import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings

# Ignorer les avertissements pour une meilleure lisibilité de la sortie
warnings.filterwarnings('ignore')

# =========================================
# Étape 1 : Préparation et nettoyage des données
# =========================================
# Demander à l'utilisateur de choisir son fichier CSV
# Note: Dans un environnement de script pur, cela nécessiterait une saisie manuelle
# ou l'utilisation d'une bibliothèque d'interface utilisateur (comme Tkinter) pour une boîte de dialogue de fichier.
csv_file_path = input("Veuillez entrer le chemin complet de votre fichier CSV (ex: C:/Users/VotreNom/Desktop/mon_fichier.csv): ")

try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Erreur: Le fichier '{csv_file_path}' n'a pas été trouvé.")
    print("Veuillez vous assurer que le chemin est correct et que le fichier existe.")
    exit() # Quitte le script si le fichier n'est pas trouvé
df = df.drop(columns=["Unnamed: 20", "Unnamed: 21"], errors='ignore')
df = df.dropna(subset=["Year Test"])
df["Year Test"] = df["Year Test"].astype(int)

colonnes_a_verifier = [
    'MFG', 'KV', 'MVA', 'Age', 'Year Test', 'O2', 'N2', 'CO2', 'CO',
    'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'C3H6', 'C3H8', 'TCG', 'TEMP', 'WATER'
]
df = df[~df[colonnes_a_verifier].eq(0.0).all(axis=1)]
df["year_num"] = df["Year Test"] - df["Year Test"].min() # Utilisé pour les features temporelles

# Ratios
for ratio, num, denom in zip(["R1", "R2", "R3", "R4", "R5"], ["CH4", "C2H2", "C2H2", "C2H6", "C2H4"], ["H2", "C2H4", "CH4", "C2H2", "C2H6"]):
    df[ratio] = df[num] / df[denom].replace(0, np.nan)
df["Duval_Total"] = df[["CH4", "C2H2", "C2H4"]].sum(axis=1).replace(0, np.nan)

# =========================================
# Étape 2 : Méthodes de classification classiques
# =========================================
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

# =========================================
# Étape 3 : Calcul coordonnées Duval (Cx, Cy)
# =========================================
gases = ["CH4", "C2H2", "C2H4", "H2", "C2H6"]
df["Total_DP"] = df[gases].sum(axis=1)
df = df[df["Total_DP"] > 0]
for g in gases:
    df[f"%{g}"] = 100 * df[g] / df["Total_DP"]
df["Cx"] = (df["%CH4"] + 0.5 * df["%C2H6"] - 0.5 * df["%C2H2"] - df["%C2H4"]) / 100
df["Cy"] = (0.866 * df["%C2H6"] + 0.866 * df["%C2H2"]) / 100

# =========================================
# Étape 4 : Encodage et préparation cible
# =========================================
method_columns = ["IRM_fault", "DRM_fault", "RRM_fault", "DTM_fault", "DPM_fault"]
for col in method_columns:
    df[f"{col}_enc"] = LabelEncoder().fit_transform(df[col])

df["true_fault"] = df[method_columns].mode(axis=1)[0]
le_target = LabelEncoder()
df["true_fault_index"] = le_target.fit_transform(df["true_fault"])

# Convertir les index des classes en labels pour référence
fault_labels = {i: c for i, c in enumerate(le_target.classes_)}


# =========================================
# Préparation des données pour la prédiction temporelle
# =========================================

# Caractéristiques temporelles
df["year_sin"] = np.sin(2 * np.pi * df["Year Test"] / 12)
df["year_cos"] = np.cos(2 * np.pi * df["Year Test"] / 12)

# Les caractéristiques pour les modèles de prédiction temporelle seront uniquement les features temporelles
X_temporal = df[["year_num", "year_sin", "year_cos"]]
y_temporal = df["true_fault_index"].values

# Années futures pour la prédiction
future_years = np.arange(df["Year Test"].min(), 2031)
future_year_num = future_years - df["Year Test"].min()
future_year_sin = np.sin(2 * np.pi * future_years / 12)
future_year_cos = np.cos(2 * np.pi * future_years / 12)

X_future_temporal = pd.DataFrame({
    "year_num": future_year_num,
    "year_sin": future_year_sin,
    "year_cos": future_year_cos
})

# Obtenir les proportions réelles pour le plotting (maintenu pour le context de Dash)
real_proportions_by_year = df.groupby("Year Test")["true_fault_index"].value_counts(normalize=True).unstack(fill_value=0)
# Renommer les colonnes pour qu'elles correspondent aux labels originaux
real_proportions_by_year.columns = [fault_labels[idx] for idx in real_proportions_by_year.columns]

# =========================================
# Définition des modèles et de leurs hyperparamètres pour GridSearchCV
# =========================================

# Grilles d'hyperparamètres pour les modèles
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    },
    "Naive Bayes": {
        # GaussianNB n'a pas d'hyperparamètres majeurs à ajuster via GridSearchCV
        # On pourrait tester 'var_smoothing' mais souvent pas nécessaire pour des gains importants
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    },
    "Logistic Regression": {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2']
    },
    "Decision Tree": {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }
}

# Initialisation des modèles pour l'évaluation avec les meilleurs hyperparamètres
models_to_evaluate = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}


# Fonction d'évaluation des métriques (identique)
def evaluate_model_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    num_classes = len(fault_labels)

    for i in range(num_classes):
        # Vérifie si la classe 'i' est présente dans la matrice de confusion calculée pour ce fold
        # La matrice de confusion peut être plus petite si toutes les classes ne sont pas présentes dans le fold.
        if i < cm.shape[0] and i < cm.shape[1]:
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FN + FP)
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        else:
            specificity = 0 # Si la classe n'est pas dans le fold, sa spécificité est considérée comme 0 pour cette moyenne
        specificity_per_class.append(specificity)
    specificity = np.mean(specificity_per_class)

    return acc, prec, rec, specificity, f1

# =========================================
# Modélisation et évaluation (sur features classiques avec GridSearchCV)
# =========================================

print("\n--- Optimisation et évaluation des modèles sur les features classiques ---")

cv_initial = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_initial = {}
best_initial_models = {} # Pour stocker les meilleurs modèles entraînés avec les hyperparamètres optimaux

X_initial = df[[f"{col}_enc" for col in method_columns] + ["Cx", "Cy"]]
y_initial = df["true_fault_index"]

# Appliquer la mise à l'échelle pour les modèles sensibles
scaler_initial = StandardScaler()
# Scaler seulement si X_initial n'est pas vide
if not X_initial.empty:
    X_initial_scaled = scaler_initial.fit_transform(X_initial)
    X_initial_scaled_df = pd.DataFrame(X_initial_scaled, columns=X_initial.columns, index=X_initial.index)
else:
    X_initial_scaled_df = X_initial # Garde un DataFrame vide si initialement vide

for name, model_instance in models_to_evaluate.items():
    print(f"\nRecherche d'hyperparamètres pour {name} (features classiques)...")
    
    # Utiliser les données mises à l'échelle si le modèle est sensible
    X_data_for_grid = X_initial_scaled_df if name in ["SVM", "Logistic Regression", "KNN"] else X_initial

    if name in param_grids and param_grids[name]:
        grid_search = GridSearchCV(
            estimator=model_instance,
            param_grid=param_grids[name],
            cv=cv_initial,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        try:
            # Vérifier si l'ensemble de données est suffisamment grand pour GridSearchCV
            if len(X_data_for_grid) > 0 and len(np.unique(y_initial)) > 1:
                grid_search.fit(X_data_for_grid, y_initial)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Meilleurs hyperparamètres pour {name}: {best_params}")
            else:
                raise ValueError("Données insuffisantes ou trop peu de classes pour GridSearchCV.")
        except ValueError as e:
            print(f"Erreur lors de GridSearchCV pour {name}: {e}. Utilisation des paramètres par défaut.")
            best_model = model_instance
            best_params = "N/A (Erreur GridSearchCV / paramètres par défaut)"
    else:
        best_model = model_instance
        best_params = "N/A (paramètres par défaut)"
        print(f"Pas de grille d'hyperparamètres définie pour {name}. Utilisation des paramètres par défaut.")
    
    best_initial_models[name] = best_model # Stocker le modèle optimisé

    # Évaluer le modèle avec les meilleurs hyperparamètres via cross_val_score
    if len(X_data_for_grid) > 0 and len(np.unique(y_initial)) > 1:
        acc = cross_val_score(best_model, X_data_for_grid, y_initial, cv=cv_initial, scoring='accuracy').mean()
        prec = cross_val_score(best_model, X_data_for_grid, y_initial, cv=cv_initial, scoring='precision_weighted').mean()
        rec = cross_val_score(best_model, X_data_for_grid, y_initial, cv=cv_initial, scoring='recall_weighted').mean()
        f1 = cross_val_score(best_model, X_data_for_grid, y_initial, cv=cv_initial, scoring='f1_weighted').mean()

        results_initial[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "Best Params": best_params}
    else:
        print(f"Données insuffisantes pour évaluer le modèle {name} avec cross_val_score.")
        results_initial[name] = {
            "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan,
            "F1": np.nan, "Best Params": best_params
        }


metrics_df_initial = pd.DataFrame(results_initial).T.round(3)
metrics_df_initial.sort_values("Accuracy", ascending=False, inplace=True)
print("\n--- Performances des modèles sur features classiques (avec hyperparamètres optimisés) ---")
print(metrics_df_initial)


# --- Vérification du surapprentissage pour les modèles classiques optimisés ---
print("\n--- Vérification du surapprentissage pour les modèles classiques optimisés ---")
for name, best_model in best_initial_models.items():
    X_data_for_pred = X_initial_scaled_df if name in ["SVM", "Logistic Regression", "KNN"] else X_initial
    
    if not X_data_for_pred.empty and len(np.unique(y_initial)) >= 2:
        try:
            best_model.fit(X_data_for_pred, y_initial)
            y_pred_train = best_model.predict(X_data_for_pred)
            train_accuracy = accuracy_score(y_initial, y_pred_train)
            
            # Récupérer l'accuracy moyenne de validation croisée déjà calculée
            cv_accuracy = results_initial[name]['Accuracy']
            
            print(f"\nModèle: {name}")
            print(f"  Accuracy sur l'ensemble d'entraînement (données vues): {train_accuracy:.3f}")
            print(f"  Accuracy moyenne de CV (données non vues): {cv_accuracy:.3f}")
            
            # Un seuil de 5% est un bon point de départ, mais peut être ajusté.
            if not pd.isna(cv_accuracy) and train_accuracy > cv_accuracy + 0.05:
                print("  --> Potentiel surapprentissage détecté : La performance sur les données d'entraînement est significativement plus élevée.")
            else:
                print("  --> Pas de signe majeur de surapprentissage (performance sur l'entraînement et CV sont proches).")
        except Exception as e:
            print(f"\nModèle: {name} - Erreur lors de la vérification du surapprentissage: {e}")
    else:
        print(f"\nModèle: {name} - Données insuffisantes ou trop peu de classes pour vérifier le surapprentissage.")


# =========================================
# Visualisations des données et des performances initiales (sections supprimées)
# =========================================

# =========================================
# Implémentation de la prédiction temporelle avec modèles ML (avec GridSearchCV)
# =========================================

print("\n--- Optimisation et évaluation des modèles temporels ---")

# Normalisation des features temporelles
scaler_temporal = StandardScaler()
if not X_temporal.empty:
    X_temporal_scaled = scaler_temporal.fit_transform(X_temporal)
    X_temporal_scaled_df = pd.DataFrame(X_temporal_scaled, columns=X_temporal.columns, index=X_temporal.index)
else:
    X_temporal_scaled_df = X_temporal

tscv = TimeSeriesSplit(n_splits=5)
results_temporal = {}
all_model_predictions = {}
best_temporal_models = {} # Pour stocker les meilleurs modèles temporels entraînés

for name, model_instance in models_to_evaluate.items():
    print(f"\nRecherche d'hyperparamètres pour {name} (features temporelles)...")
    
    # Utiliser les données mises à l'échelle si le modèle est sensible
    X_data_for_grid = X_temporal_scaled_df if name in ["SVM", "Logistic Regression", "KNN"] else X_temporal

    if name in param_grids and param_grids[name]:
        grid_search = GridSearchCV(
            estimator=model_instance,
            param_grid=param_grids[name],
            cv=tscv, # Utilisation de TimeSeriesSplit ici
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        try:
            if len(X_data_for_grid) > 0 and len(np.unique(y_temporal)) > 1:
                grid_search.fit(X_data_for_grid, y_temporal)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Meilleurs hyperparamètres pour {name}: {best_params}")
            else:
                raise ValueError("Données insuffisantes ou trop peu de classes pour GridSearchCV.")
        except ValueError as e:
            print(f"Erreur lors de GridSearchCV pour {name}: {e}. Utilisation des paramètres par défaut.")
            best_model = model_instance
            best_params = "N/A (Erreur GridSearchCV / paramètres par défaut)"
    else:
        best_model = model_instance
        best_params = "N/A (paramètres par défaut)"
        print(f"Pas de grille d'hyperparamètres définie pour {name}. Utilisation des paramètres par défaut.")
    
    best_temporal_models[name] = best_model # Stocker le modèle optimisé

    accs, precs, recs, specs, f1s = [], [], [], [], []

    # Évaluation du modèle avec les meilleurs hyperparamètres via TimeSeriesSplit
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_data_for_grid)):
        X_train_temp, X_test_temp = X_data_for_grid.iloc[train_idx], X_data_for_grid.iloc[test_idx]
        y_train_temp, y_test_temp = y_temporal[train_idx], y_temporal[test_idx]
        
        if len(np.unique(y_train_temp)) < 2 or len(np.unique(y_test_temp)) < 2:
            print(f"  Fold {fold_idx}: Moins de 2 classes dans les données d'entraînement/test. Saut du fold.")
            continue
        try:
            best_model.fit(X_train_temp, y_train_temp)
            y_pred_temp = best_model.predict(X_test_temp)
            
            acc, prec, rec, spec, f1 = evaluate_model_metrics(y_test_temp, y_pred_temp)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            specs.append(spec)
            f1s.append(f1)
        except Exception as e:
            print(f"  Erreur lors de l'entraînement ou de la prédiction du fold {fold_idx} pour {name}: {e}. Saut du fold.")


    if accs:
        results_temporal[name] = {
            "Accuracy": np.mean(accs),
            "Precision": np.mean(precs),
            "Recall": np.mean(recs),
            "Specificity": np.mean(specs),
            "F1": np.mean(f1s),
            "Best Params": best_params
        }

        print(f"Mean CV Accuracy   : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"Mean CV Precision  : {np.mean(precs):.3f}")
        print(f"Mean CV Recall     : {np.mean(recs):.3f}")
        print(f"Mean CV Specificity: {np.mean(specs):.3f}")
        print(f"Mean CV F1-Score   : {np.mean(f1s):.3f}")
    else:
        print(f"Aucun fold évalué pour le modèle {name} en raison d'un nombre insuffisant de classes ou d'erreurs.")
        results_temporal[name] = {
            "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan,
            "Specificity": np.nan, "F1": np.nan, "Best Params": best_params
        }

    # Entraîner le modèle optimisé sur l'ensemble des données temporelles pour la prédiction future
    if len(np.unique(y_temporal)) >= 2:
        try:
            best_model.fit(X_data_for_grid, y_temporal) # Utiliser les données scalées complètes
            if hasattr(best_model, "predict_proba"):
                X_future_temporal_scaled = scaler_temporal.transform(X_future_temporal) if name in ["SVM", "Logistic Regression", "KNN"] else X_future_temporal
                
                proba_future = best_model.predict_proba(X_future_temporal_scaled)
                
                predicted_proportions_future_df = pd.DataFrame(0.0, index=future_years, columns=[fault_labels[i] for i in range(len(fault_labels))])
                
                model_classes = best_model.classes_
                for i, class_idx in enumerate(model_classes):
                    label = fault_labels[class_idx]
                    if label in predicted_proportions_future_df.columns:
                        predicted_proportions_future_df[label] = proba_future[:, i]
                
                all_model_predictions[name] = predicted_proportions_future_df
            else:
                print(f"Attention: Le modèle {name} n'a pas de méthode predict_proba. La prédiction détaillée des proportions est ignorée.")
        except Exception as e:
            print(f"Erreur lors de l'entraînement final ou de la prédiction future pour {name}: {e}. La prédiction future est ignorée.")
    else:
        print(f"Attention: Pas assez de classes uniques ({len(np.unique(y_temporal))}) dans X_temporal pour entraîner le modèle {name} pour la prédiction future.")


# Convertir les résultats temporels en DataFrame pour le plotting
metrics_df_temporal = pd.DataFrame(results_temporal).T.round(3)
metrics_df_temporal.sort_values("Accuracy", ascending=False, inplace=True)
print("\n--- Performances des modèles sur features temporelles (avec hyperparamètres optimisés) ---")
print(metrics_df_temporal)

# --- Vérification du surapprentissage pour les modèles temporels optimisés ---
print("\n--- Vérification du surapprentissage pour les modèles temporels optimisés ---")
for name, best_model in best_temporal_models.items():
    X_data_for_pred = X_temporal_scaled_df if name in ["SVM", "Logistic Regression", "KNN"] else X_temporal
    
    if not X_data_for_pred.empty and len(np.unique(y_temporal)) >= 2:
        try:
            best_model.fit(X_data_for_pred, y_temporal)
            y_pred_train = best_model.predict(X_data_for_pred)
            train_accuracy = accuracy_score(y_temporal, y_pred_train)
            
            cv_accuracy = results_temporal[name]['Accuracy']
            
            print(f"\nModèle: {name}")
            print(f"  Accuracy sur l'ensemble d'entraînement (données vues): {train_accuracy:.3f}")
            print(f"  Accuracy moyenne de CV (données non vues): {cv_accuracy:.3f}")
            
            if not pd.isna(cv_accuracy) and train_accuracy > cv_accuracy + 0.05:
                print("  --> Potentiel surapprentissage détecté : La performance sur les données d'entraînement est significativement plus élevée.")
            else:
                print("  --> Pas de signe majeur de surapprentissage (performance sur l'entraînement et CV sont proches).")
        except Exception as e:
            print(f"\nModèle: {name} - Erreur lors de la vérification du surapprentissage: {e}")
    else:
        print(f"\nModèle: {name} - Données insuffisantes ou trop peu de classes pour vérifier le surapprentissage.")

print("\n--- Script terminé ---")

import joblib

# Déterminez le meilleur modèle temporel à sauvegarder
# Si metrics_df_temporal est vide ou que toutes les accuracies sont NaN, choisissez un par défaut ou gérez l'erreur.
if not metrics_df_temporal.empty and not metrics_df_temporal['Accuracy'].isnull().all():
    best_temporal_model_name = metrics_df_temporal['Accuracy'].idxmax()
else:
    best_temporal_model_name = list(models_to_evaluate.keys())[0] # Ou gérer l'erreur comme nécessaire

joblib.dump({
    "df": df,
    "real_proportions_by_year": real_proportions_by_year,
    "all_model_predictions": all_model_predictions,
    "metrics_df_temporal": metrics_df_temporal,
    "fault_labels": fault_labels,
    "best_temporal_model_name": best_temporal_model_name
}, "dash_context.pkl")
