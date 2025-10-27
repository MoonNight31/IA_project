"""
Script adaptatif pour l'analyse de performances automobiles.
Sélectionne automatiquement le meilleur framework (sklearn ou TensorFlow) selon la taille du dataset.
"""

import sys
from pathlib import Path

# Ajouter le dossier src au PYTHONPATH
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import time

from project_ia.model_sklearn import SkLearnCarNN
from project_ia.model_tensorflow import TensorFlowCarNN


class AdaptiveCarNN:
    """
    Wrapper adaptatif qui sélectionne automatiquement le meilleur framework
    (sklearn ou TensorFlow) selon la taille du dataset.
    """
    
    def __init__(self, framework: str = "auto"):
        """
        Args:
            framework: "sklearn", "tensorflow", ou "auto" (choix automatique)
        """
        self.framework = framework
        self.model_impl = None  # Instance de SkLearnCarNN ou TensorFlowCarNN
        
    def detect_optimal_framework(self, n_samples: int) -> str:
        """Détecte le meilleur framework selon la taille du dataset."""
        if n_samples < 50000:
            return "sklearn"  # Plus rapide pour petits datasets
        else:
            try:
                import tensorflow
                return "tensorflow"  # Meilleur pour gros datasets
            except ImportError:
                return "sklearn"


def main():
    """Fonction principale avec adaptation automatique."""
    print("=" * 80)
    print("  ANALYSE ADAPTATIVE - Reseaux de Neurones pour Performances Automobiles")
    print("=" * 80)
    
    csv_path = Path("cars.csv")
    
    if not csv_path.exists():
        print("\nErreur: Fichier cars.csv introuvable.")
        return
    
    # Initialisation du wrapper adaptatif
    adaptive = AdaptiveCarNN(framework="auto")
    
    # Chargement rapide pour détecter la taille
    print("Chargement du dataset...")
    start_time = time.time()
    df = pd.read_csv(csv_path)
    df = df.dropna()
    if 'is_started' in df.columns:
        df = df.drop('is_started', axis=1)
    load_time = time.time() - start_time
    
    print(f"  Dataset charge en {load_time:.2f}s")
    print(f"  Taille: {len(df):,} voitures")
    print(f"  Memoire: ~{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Détection du framework optimal
    optimal_framework = adaptive.detect_optimal_framework(len(df))
    print(f"\nFramework optimal detecte: {optimal_framework.upper()}")
    
    # Initialisation du modèle spécifique
    if optimal_framework == "sklearn":
        model = SkLearnCarNN()
    else:
        model = TensorFlowCarNN()
    
    adaptive.model_impl = model
    adaptive.framework = optimal_framework
    
    # Préparation des données
    X_train, X_test, y_train, y_test = model.prepare_regression_data(df)
    
    # Construction du modèle
    if optimal_framework == "sklearn":
        # sklearn ne nécessite pas de construction explicite
        pass
    else:
        model.build_regression_model(input_dim=X_train.shape[1])
    
    # Entraînement
    print("\n" + "=" * 80)
    train_time = model.train(X_train, y_train, X_test, y_test)
    
    # Évaluation
    print("=" * 80)
    mae, rmse, r2, predictions = model.evaluate_regression(X_test, y_test)
    
    # Résultats
    print("\n" + "=" * 80)
    print("  RESULTATS FINAUX")
    print("=" * 80)
    print(f"\nMetriques de performance:")
    print(f"  MAE  : {mae:.3f} secondes")
    print(f"  RMSE : {rmse:.3f} secondes")
    print(f"  R²   : {r2:.3f} ({r2*100:.1f}%)")
    
    print(f"\nPerformances d'execution:")
    print(f"  Temps d'entrainement: {train_time:.2f}s")
    print(f"  Vitesse: {len(X_train) / train_time:.0f} echantillons/s")
    
    # Exemples
    print(f"\nExemples de predictions (10 premiers):")
    for i in range(min(10, len(y_test))):
        diff = abs(y_test[i] - predictions[i])
        symbol = "OK" if diff < 0.5 else "~" if diff < 1.0 else "X"
        print(f"  [{symbol}] Reel: {y_test[i]:5.2f}s | Predit: {predictions[i]:5.2f}s | Ecart: {diff:4.2f}s")
    
    # Visualisation
    print(f"\nGeneration des graphiques...")
    model.plot_regression_results(y_test, predictions, save_path=f"adaptive_{optimal_framework}_results.png")
    
    print("\n" + "=" * 80)
    print("  ANALYSE TERMINEE")
    print("=" * 80)


if __name__ == "__main__":
    main()
