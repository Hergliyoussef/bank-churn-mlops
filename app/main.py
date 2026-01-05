import os
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.fastapi.fastapi_middleware import FastAPIMiddleware

# 1. Définition du schéma de données (Entrée de l'IA)
class CustomerData(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_Germany: int
    Geography_Spain: int
    Gender_Male: int

# 2. Initialisation de l'API
app = FastAPI(title="Bank Churn Prediction API")

# 3. Configuration du Monitoring (Module 6)
# Récupération de la clé injectée dans les variables d'environnement Azure
connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
logger = logging.getLogger(__name__)

if connection_string:
    # Envoie les logs vers Azure Application Insights
    logger.addHandler(AzureLogHandler(connection_string=connection_string))
    # Middleware pour suivre automatiquement les performances des requêtes
    app.add_middleware(FastAPIMiddleware)
    logger.info("Application Insights configuré avec succès.")
else:
    logger.warning("APPLICATIONINSIGHTS_CONNECTION_STRING non trouvée. Monitoring local uniquement.")

# 4. Chargement du modèle au démarrage
MODEL_PATH = "model/model.joblib"
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Modèle chargé depuis {MODEL_PATH}")
        else:
            logger.error(f"Fichier modèle introuvable à {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {str(e)}")

# 5. Endpoints de l'API

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction du Churn Bancaire"}

@app.get("/health")
def health_check():
    """Vérifie si le modèle est bien chargé"""
    if model is None:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(data: CustomerData):
    """Effectue une prédiction et log les données pour le Data Drift"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        # Conversion des données en DataFrame pour le modèle
        input_df = pd.DataFrame([data.dict()])
        
        # Log des données entrantes pour surveillance future du Drift
        logger.info(f"Requête de prédiction reçue : {data.dict()}")
        
        # Prédiction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]
        
        result = {
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(probability[0])
        }
        
        # Log du résultat
        logger.info(f"Résultat de la prédiction : {result}")
        
        return result

    except Exception as e:
        logger.error(f"Erreur pendant la prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))