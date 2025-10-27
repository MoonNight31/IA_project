"""
Script adaptatif pour l'analyse de performances automobiles.
Sélectionne automatiquement le meilleur framework (sklearn ou TensorFlow) selon la taille du dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Ajouter le dossier src au PYTHONPATH
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Imports avec gestion d'erreur
try:
    from sklearn.metrics import mean_squared_error, r2_score
    sklearn_available = True
except ImportError:
    print("⚠ scikit-learn non disponible")
    sklearn_available = False

try:
    import tensorflow as tf
    tensorflow_available = True
except ImportError:
    print("⚠ TensorFlow non disponible")
    tensorflow_available = False

# Imports des modèles avec gestion d'erreur
model_sklearn = None
model_tensorflow = None

try:
    from project_ia.model_sklearn import SkLearnCarNN
    model_sklearn = SkLearnCarNN
    print("✓ Modèle sklearn importé")
except ImportError as e:
    print(f"⚠ Impossible d'importer model_sklearn: {e}")

try:
    from project_ia.model_tensorflow import TensorFlowCarNN
    model_tensorflow = TensorFlowCarNN
    print("✓ Modèle TensorFlow importé")
except ImportError as e:
    print(f"⚠ Impossible d'importer model_tensorflow: {e}")

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
        print(f"\nDétection du framework optimal pour {n_samples:,} échantillons...")
        
        # Vérifier la disponibilité des frameworks
        available_frameworks = []
        
        if sklearn_available and model_sklearn:
            available_frameworks.append("sklearn")
            print("  ✓ sklearn disponible")
        
        if tensorflow_available and model_tensorflow:
            available_frameworks.append("tensorflow")
            print("  ✓ TensorFlow disponible")
        
        if not available_frameworks:
            raise ImportError("Aucun framework disponible. Installez sklearn ou tensorflow.")
        
        # Logique de sélection
        if len(available_frameworks) == 1:
            selected = available_frameworks[0]
            print(f"  → Seul framework disponible: {selected}")
        elif n_samples < 50000:
            selected = "sklearn" if "sklearn" in available_frameworks else "tensorflow"
            print(f"  → Dataset petit ({n_samples:,} < 50k): {selected} recommandé")
        else:
            selected = "tensorflow" if "tensorflow" in available_frameworks else "sklearn"
            print(f"  → Dataset large ({n_samples:,} ≥ 50k): {selected} recommandé")
        
        return selected

def load_and_validate_data(csv_path: Path) -> pd.DataFrame:
    """Charge et valide le dataset."""
    print("Chargement du dataset...")
    start_time = time.time()
    
    try:
        # Essayer différents encodages
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='latin-1')
            except:
                df = pd.read_csv(csv_path, encoding='cp1252')
        
        load_time = time.time() - start_time
        
        print(f"  ✓ Dataset chargé en {load_time:.2f}s")
        print(f"  Forme initiale: {df.shape}")
        
        # Nettoyage
        initial_size = len(df)
        df = df.dropna()
        cleaned_size = len(df)
        
        if 'is_started' in df.columns:
            df = df.drop('is_started', axis=1)
        
        # Vérifications
        required_cols = ['power', 'weight', 'zero_to_hundred']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        print(f"  ✓ Nettoyage: {initial_size:,} → {cleaned_size:,} lignes")
        print(f"  ✓ Colonnes: {list(df.columns)}")
        print(f"  ✓ Mémoire: ~{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Erreur lors du chargement: {e}")
        raise

def main():
    """Fonction principale avec adaptation automatique."""
    print("=" * 80)
    print("  ANALYSE ADAPTATIVE - Réseaux de Neurones pour Performances Automobiles")
    print("=" * 80)
    
    csv_path = Path("cars.csv")
    
    if not csv_path.exists():
        print(f"\n✗ Erreur: Fichier {csv_path} introuvable.")
        print("Solutions:")
        print("1. Exécutez program.py pour générer le dataset")
        print("2. Vérifiez que vous êtes dans le bon répertoire")
        return
    
    try:
        # Chargement et validation
        df = load_and_validate_data(csv_path)
        
        # Initialisation du wrapper adaptatif
        adaptive = AdaptiveCarNN(framework="auto")
        
        # Détection du framework optimal
        optimal_framework = adaptive.detect_optimal_framework(len(df))
        
        # Initialisation du modèle spécifique
        print(f"\nInitialisation du modèle {optimal_framework.upper()}...")
        
        if optimal_framework == "sklearn":
            if not model_sklearn:
                raise ImportError("sklearn non disponible")
            model = model_sklearn()
        else:
            if not model_tensorflow:
                raise ImportError("TensorFlow non disponible")
            model = model_tensorflow()
        
        adaptive.model_impl = model
        adaptive.framework = optimal_framework
        
        # Préparation des données
        print("\nPréparation des données...")
        X_train, X_test, y_train, y_test = model.prepare_regression_data(df)
        
        # Construction du modèle (TensorFlow uniquement)
        if optimal_framework == "tensorflow":
            print("Construction du modèle TensorFlow...")
            model.build_regression_model(input_dim=X_train.shape[1])
            print(f"✓ Modèle construit avec {X_train.shape[1]} features")
        
        # Entraînement adaptatif selon le framework
        print("\n" + "=" * 80)
        print("ENTRAÎNEMENT")
        print("=" * 80)
        
        if optimal_framework == "sklearn":
            # Vérifier si train_and_evaluate existe, sinon utiliser les méthodes séparées
            if hasattr(model, 'train_and_evaluate'):
                train_time, mae, rmse, r2, predictions = model.train_and_evaluate(X_train, y_train, X_test, y_test)
            else:
                # Fallback: utiliser les méthodes séparées
                train_time = model.train_regression(X_train, y_train)
                print("\n" + "=" * 80)
                print("ÉVALUATION")
                print("=" * 80)
                mae, rmse, r2, predictions = model.evaluate_regression(X_test, y_test)
        else:
            # TensorFlowCarNN utilise train puis evaluate_regression
            train_time = model.train(X_train, y_train, X_test, y_test)
            # Évaluation
            print("\n" + "=" * 80)
            print("ÉVALUATION")
            print("=" * 80)
            mae, rmse, r2, predictions = model.evaluate_regression(X_test, y_test)
        
        # Résultats
        print("\n" + "=" * 80)
        print("  RÉSULTATS FINAUX")
        print("=" * 80)
        print(f"\nFramework utilisé: {optimal_framework.upper()}")
        print(f"Dataset: {len(df):,} voitures")
        
        print(f"\nMétriques de performance:")
        print(f"  MAE  : {mae:.3f} secondes")
        print(f"  RMSE : {rmse:.3f} secondes")
        print(f"  R²   : {r2:.3f} ({r2*100:.1f}%)")
        
        print(f"\nPerformances d'exécution:")
        print(f"  Temps d'entraînement: {train_time:.2f}s")
        print(f"  Vitesse: {len(X_train) / train_time:.0f} échantillons/s")
        
        # Exemples de prédictions
        print(f"\nExemples de prédictions (10 premiers):")
        for i in range(min(10, len(y_test))):
            diff = abs(y_test[i] - predictions[i])
            symbol = "✓" if diff < 0.5 else "~" if diff < 1.0 else "✗"
            print(f"  [{symbol}] Réel: {y_test[i]:5.2f}s | Prédit: {predictions[i]:5.2f}s | Écart: {diff:4.2f}s")
        
        # Visualisation
        print(f"\nGénération des graphiques...")
        try:
            save_path = f"adaptive_{optimal_framework}_results.png"
            model.plot_regression_results(y_test, predictions, save_path=save_path)
            print(f"✓ Graphiques sauvegardés: {save_path}")
        except Exception as e:
            print(f"⚠ Erreur lors de la visualisation: {e}")
        
        print("\n" + "=" * 80)
        print("  ANALYSE TERMINÉE AVEC SUCCÈS")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interruption utilisateur")
        
    except Exception as e:
        print(f"\n✗ Erreur critique: {e}")
        print("\nDiagnostic:")
        import traceback
        traceback.print_exc()
        
        print("\nSolutions possibles:")
        print("1. Vérifiez l'installation: python -m pip install scikit-learn tensorflow")
        print("2. Vérifiez que cars.csv existe et est valide")
        print("3. Exécutez diagnostic.py pour plus d'informations")

if __name__ == "__main__":
    main()
