import pandas as pd

# Création du DataFrame à partir des données de l'image
data = {
    "Instance": list(range(1, 11)),
    "Color": [0.7, 2.3, 4.5, 2.9, 3.4, 1, 2, 2, 4, 4],
    "Viscosity": [10.23, 10.87, 11.19, 10.97, 11.5, 12.45, 10, 12.3, 12, 12],
    "Acid": [0.012, 0.091, 0.42, 0.021, 0.07, 0.046, 0.106, 0.062, 0.08, 0.3],
    "Dielectric": [57, 22, 30, 65, 57, 56, 50, 52, 48, 30],
    "TgTo": [0.072, 0.019, 0.55, 0.018, 0.105, 0.15, 0.025, 0.020, 0.022, 0.6],
    "MC_Moisture_Content": [16, 40, 42, 7, 23, 31, 35, 32, 37, 42]
}

df = pd.DataFrame(data)

# Fonction de classification
def classify(row):
    c, v, a, d, t, m = row["Color"], row["Viscosity"], row["Acid"], row["Dielectric"], row["TgTo"], row["MC_Moisture_Content"]
    
    if 1.4 <= c <= 3.8 and 9.9 <= v <= 10.2 and 0.044 <= a <= 0.11 and 55 <= d <= 65 and 0.018 <= t <= 0.04 and 7 <= m <= 24:
        return "KEEP"
    elif 1.4 <= c <= 3.8 and 10 <= v <= 16.1 and 0.019 <= a <= 0.1 and 30 <= d <= 55 and 0.019 <= t <= 0.067 and 19 <= m <= 40:
        return "FILTER"
    elif 2.6 <= c <= 4.8 and 10.1 <= v <= 10.2 and 0.049 <= a <= 0.15 and 39 <= d <= 67 and 0.029 <= t <= 0.09 and 9 <= m <= 40:
        return "RECLAIM"
    elif 4 <= c <= 5 and 10.1 <= v <= 10.2 and 0.075 <= a <= 0.32 and 18 <= d <= 54 and 0.07 <= t <= 0.11 and 22 <= m <= 45:
        return "DISCARD"
    else:
        return "UNDEFINED"

# Application de la fonction
df["Classification"] = df.apply(classify, axis=1)
print(df[["Instance", "Classification"]])

