import pandas as pd

# === Données ===
data = {
    'Color': [0.7, 2.3, 4.5, 2.9, 3.4, 1.0, 2.0, 2.0, 4.0, 4.0],
    'Viscosity': [10.23, 10.87, 11.19, 10.97, 11.5, 12.45, 10.0, 12.3, 12.0, 12.0],
    'Acidity': [0.012, 0.091, 0.42, 0.021, 0.07, 0.046, 0.106, 0.062, 0.08, 0.3],
    'Dielectric Strength': [57, 22, 0, 65, 57, 56, 40, 52, 48, 30],
    'Tgδ': [0.072, 0.019, 0.55, 0.018, 0.105, 0.15, 0.025, 0.020, 0.022, 0.6],
    'Water Content': [16, 40, 42, 7, 23, 31, 35, 32, 33, 42],
    'Expected': ['Keep', 'Filter', 'Discard', 'Keep', 'Reclaim',
                 'Keep', 'Filter', 'Keep', 'Reclaim', 'Discard']
}
df = pd.DataFrame(data)

# === Seuils ===
thresholds = {
    'Color': 2,
    'Viscosity': 10.5,
    'Acidity': 0.1,
    'Dielectric Strength': 40,
    'Tgδ': 0.3,
    'Water Content': 30
}

# === Matrice de corrélation et pondération ===
correlation_matrix = df.drop(columns='Expected').corr().abs()
raw_weights = {p: 1 - correlation_matrix[p].drop(p).sum() for p in thresholds}
weight_total = sum(raw_weights.values())
weights = {k: v / weight_total for k, v in raw_weights.items()}

# === Fonction de gravité ===
def severity_score(value, threshold, is_minimum=False):
    if is_minimum:
        if value >= threshold:
            return 0
        elif value >= threshold * 0.75:
            return 1
        elif value >= threshold * 0.5:
            return 2
        else:
            return 3
    else:
        if value <= threshold:
            return 0
        elif value <= threshold * 1.25:
            return 1
        elif value <= threshold * 1.5:
            return 2
        else:
            return 3

# === Évaluation des échantillons ===
scores = []
decisions = []

for _, row in df.iterrows():
    score = 0
    for param in thresholds:
        is_min = (param == 'Dielectric Strength')
        grav = severity_score(row[param], thresholds[param], is_minimum=is_min)
        score += grav * weights[param]
    scores.append(round(score, 3))
    if score <= 0.3:
        decisions.append("Keep")
    elif score <= 0.43:
        decisions.append("Filter")
    elif score <= 0.9:
        decisions.append("Reclaim")
    else:
        decisions.append("Discard")

# Résultat
df['Score'] = scores
df['Predicted'] = decisions

# Affichage
print(df[['Color', 'Viscosity', 'Acidity', 'Dielectric Strength', 'Tgδ', 'Water Content',
          'Expected', 'Score', 'Predicted']])
