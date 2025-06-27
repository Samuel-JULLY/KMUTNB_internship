import pandas as pd

# Données
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

def classify_best_match(row):
    c, v, a, d, t, m = row["Color"], row["Viscosity"], row["Acid"], row["Dielectric"], row["TgTo"], row["MC_Moisture_Content"]

    categories = {
        "KEEP": [
            c <= 2.7,
            v <= 11,
            a <= 0.045,
            d <= 60,
            t <= 0.032,
            m <= 17
        ],
        "FILTER": [
            c <= 2.7,
            v <= 11,
            a <= 0.06,
            d >= 44,
            t <= 0.041,
            m <= 30
        ],
        "RECLAIM": [
            c <= 3.8,
            v > 11,
            a <= 0.1,
            d >= 50,
            t <= 0.05,
            m <= 25
        ],
        "DISCARD": [
            c < 4.8,
            v < 11,
            a < 0.2,
            d < 35,
            t < 0.09,
            m < 35
        ]
    }

    match_counts = {cat: sum(conditions) for cat, conditions in categories.items()}
    best_match = max(match_counts, key=match_counts.get)
    return best_match if match_counts[best_match] > 0 else "UNDEFINED"


df["Classification"] = df.apply(classify_best_match, axis=1)
print(df[["Instance", "Classification"]])
