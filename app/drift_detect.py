import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(reference_file: str, production_file: str, threshold: float = 0.05):
    """
    Compare le dataset de référence et les données de production 
    pour détecter une dérive statistique.
    """
    try:
        df_ref = pd.read_csv(reference_file)
        df_prod = pd.read_csv(production_file)
        
        results = {}
        # On se concentre sur les colonnes numériques clés
        cols = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
        
        for col in cols:
            # Test de Kolmogorov-Smirnov pour comparer les distributions
            stat, p_value = ks_2samp(df_ref[col], df_prod[col])
            results[col] = {
                "drift_detected": bool(p_value < threshold),
                "p_value": float(p_value),
                "statistic": float(stat),
                "type": "numerical"
            }
        return results
    except Exception as e:
        print(f"Erreur lors de la détection de drift : {e}")
        return {}