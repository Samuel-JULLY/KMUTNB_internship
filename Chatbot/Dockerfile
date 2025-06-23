# Utilise une image Python officielle
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie les fichiers nécessaires
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose le port utilisé par Gradio
EXPOSE 7860

# Commande pour lancer l'app
CMD ["python", "app.py"]
