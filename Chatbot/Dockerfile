FROM python:3.11-slim

# Installe les dépendances système minimales
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Crée un dossier de travail
WORKDIR /app

# Copie les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Expose le port Gradio
EXPOSE 7860

# Démarre l’application
CMD ["python", "app.py"]
