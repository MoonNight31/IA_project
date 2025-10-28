"""
Modèle de réseau de neurones avec TensorFlow/Keras pour l'analyse de performances automobiles.
Optimisé pour les datasets de moyenne à grande taille (> 50k échantillons).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import time


class TensorFlowCarNN:
    """
    Réseau de neurones avec TensorFlow/Keras pour l'analyse automobile.
    Ajuste automatiquement les paramètres selon la taille du dataset.
    """
    
    def __init__(self):
        """Initialise le modèle TensorFlow."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            self.tf = tf
            self.keras = keras
        except ImportError:
            raise ImportError("TensorFlow n'est pas installé. Exécutez: pip install tensorflow")
        
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.dataset_size = 0
        self.config = {}
    
    def calculate_optimal_params(self, n_samples: int, n_features: int):
        """
        Calcule les hyperparamètres optimaux.
        
        Args:
            n_samples: Nombre d'échantillons
            n_features: Nombre de features
        """
        self.dataset_size = n_samples
        
        # Batch size : augmente avec la taille
        if n_samples < 10000:
            batch_size = 64
        elif n_samples < 50000:
            batch_size = 128
        elif n_samples < 200000:
            batch_size = 256
        else:
            batch_size = 512
        
        # Epochs : diminue avec la taille
        if n_samples < 10000:
            max_epochs = 100
            patience = 15
        elif n_samples < 50000:
            max_epochs = 75
            patience = 12
        else:
            max_epochs = 50
            patience = 10
        
        # Architecture
        first_layer = min(256, max(64, n_features * 4))
        if n_samples < 10000:
            hidden_layers = [first_layer // 2, first_layer // 4]
        else:
            hidden_layers = [first_layer, first_layer // 2, first_layer // 4]
        
        # Test size
        if n_samples < 10000:
            test_size = 0.2
        elif n_samples < 100000:
            test_size = 0.15
        else:
            test_size = 0.1
        
        self.config = {
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'patience': patience,
            'hidden_layers': tuple(hidden_layers),
            'test_size': test_size,
            'verbose': 1 if n_samples < 50000 else 0
        }
        
        return self.config
    
    def load_and_analyze(self, csv_path: Path) -> pd.DataFrame:
        """Charge et analyse le dataset."""
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
        
        return df
    
    def prepare_regression_data(self, df: pd.DataFrame, target_col: str = "zero_to_hundred"):
        """Prépare les données pour la régression."""
        print("\nPreparation des donnees pour la regression...")
        
        # Features numériques
        numeric_features = ['power', 'torque', 'weight', 'aerodynamic_level', 
                           'turbo_count', 'year', 'doors_number']
        
        # Features catégorielles
        categorical_features = ['fuel_type', 'transmission_type', 'manufacturer']
        
        df_processed = df.copy()
        
        # One-hot encoding
        encoded_features = []
        for cat_feat in categorical_features:
            if cat_feat in df.columns:
                dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat, drop_first=True)
                encoded_features.extend(dummies.columns.tolist())
                df_processed = pd.concat([df_processed, dummies], axis=1)
        
        feature_cols = numeric_features + encoded_features
        X = df_processed[feature_cols].values
        y = df[target_col].values
        
        self.feature_names = feature_cols
        n_features = len(feature_cols)
        
        # Calculer les paramètres optimaux
        config = self.calculate_optimal_params(len(X), n_features)
        
        print(f"\nConfiguration TensorFlow:")
        print(f"  Features: {n_features}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Max epochs: {config['max_epochs']}")
        print(f"  Patience: {config['patience']}")
        print(f"  Hidden layers: {config['hidden_layers']}")
        print(f"  Test size: {config['test_size']*100:.0f}%")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['test_size'], random_state=42
        )
        
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Normalisation
        print("\nNormalisation des donnees...")
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def prepare_classification_data(self, df: pd.DataFrame, target_col: str = "manufacturer"):
        """Prépare les données pour la classification."""
        print("\nPreparation des donnees pour la classification...")
        
        numeric_features = ['power', 'torque', 'weight', 'aerodynamic_level',
                           'turbo_count', 'year', 'doors_number', 'zero_to_hundred',
                           'max_speed', 'fuel_efficiency']
        
        X = df[numeric_features].values
        y = self.label_encoder.fit_transform(df[target_col])
        
        self.feature_names = numeric_features
        n_features = len(numeric_features)
        
        config = self.calculate_optimal_params(len(X), n_features)
        
        print(f"\nConfiguration TensorFlow:")
        print(f"  Features: {n_features}")
        print(f"  Classes: {len(self.label_encoder.classes_)}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Hidden layers: {config['hidden_layers']}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['test_size'], random_state=42, stratify=y
        )
        
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def build_regression_model(self, input_dim: int):
        """Construit le modèle de régression."""
        model = self.keras.Sequential()
        
        # Première couche
        model.add(self.keras.layers.Dense(
            self.config['hidden_layers'][0], 
            activation='relu', 
            input_dim=input_dim
        ))
        model.add(self.keras.layers.Dropout(0.2))
        
        # Couches cachées
        for units in self.config['hidden_layers'][1:]:
            model.add(self.keras.layers.Dense(units, activation='relu'))
            model.add(self.keras.layers.Dropout(0.2))
        
        # Couche de sortie
        model.add(self.keras.layers.Dense(1))
        
        # Compilation
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def build_classification_model(self, input_dim: int, num_classes: int):
        """Construit le modèle de classification."""
        model = self.keras.Sequential()
        
        # Première couche
        model.add(self.keras.layers.Dense(
            self.config['hidden_layers'][0], 
            activation='relu', 
            input_dim=input_dim
        ))
        model.add(self.keras.layers.Dropout(0.3))
        
        # Couches cachées
        for units in self.config['hidden_layers'][1:]:
            model.add(self.keras.layers.Dense(units, activation='relu'))
            model.add(self.keras.layers.Dropout(0.3))
        
        # Couche de sortie
        model.add(self.keras.layers.Dense(num_classes, activation='softmax'))
        
        # Compilation
        model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_test, y_test):
        """Entraîne le modèle."""
        print("\n" + "=" * 80)
        print("ENTRAINEMENT - TensorFlow/Keras")
        print("=" * 80)
        print(f"Architecture: {self.config['hidden_layers']}")
        
        # Callbacks
        early_stopping = self.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=self.config['patience'], 
            restore_best_weights=True
        )
        
        # Entraînement
        start_time = time.time()
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.config['max_epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stopping],
            verbose=self.config['verbose']
        )
        train_time = time.time() - start_time
        
        print(f"\nEntrainement termine en {train_time:.2f}s")
        print(f"Epochs executes: {len(self.history.history['loss'])}")
        
        return train_time
    
    def evaluate_regression(self, X_test, y_test):
        """Évalue le modèle de régression."""
        print("\n" + "=" * 80)
        print(f"EVALUATION sur {len(X_test):,} echantillons de test")
        print("=" * 80)
        
        predictions = self.model.predict(X_test, verbose=0).flatten()
        
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        return mae, rmse, r2, predictions
    
    def evaluate_classification(self, X_test, y_test):
        """Évalue le modèle de classification."""
        print("\n" + "=" * 80)
        print(f"EVALUATION sur {len(X_test):,} echantillons de test")
        print("=" * 80)
        
        predictions = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, y_pred
    
    def plot_regression_results(self, y_test, predictions, save_path: str = "tensorflow_regression.png"):
        """Visualise les résultats de régression."""
        errors = np.abs(y_test - predictions)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Prédictions vs Réalité
        axes[0, 0].scatter(y_test, predictions, alpha=0.3, s=1)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Valeurs reelles (s)')
        axes[0, 0].set_ylabel('Predictions (s)')
        axes[0, 0].set_title('Predictions vs Realite - TensorFlow')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution des erreurs
        axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.median(errors), color='r', linestyle='--', 
                          label=f'Mediane: {np.median(errors):.3f}s')
        axes[0, 1].set_xlabel('Erreur absolue (s)')
        axes[0, 1].set_ylabel('Frequence')
        axes[0, 1].set_title('Distribution des erreurs')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Erreur en fonction de la prédiction
        axes[1, 0].scatter(predictions, errors, alpha=0.3, s=1)
        axes[1, 0].axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='0.5s')
        axes[1, 0].axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='1.0s')
        axes[1, 0].set_xlabel('Predictions (s)')
        axes[1, 0].set_ylabel('Erreur absolue (s)')
        axes[1, 0].set_title('Erreur en fonction de la prediction')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Statistiques
        axes[1, 1].axis('off')
        stats_text = f"""
        STATISTIQUES - TensorFlow
        
        Dataset: {self.dataset_size:,} voitures
        
        Configuration:
        - Batch size: {self.config['batch_size']}
        - Architecture: {self.config['hidden_layers']}
        - Epochs: {len(self.history.history['loss'])}
        
        Erreurs:
        - MAE: {np.mean(errors):.3f}s
        - Mediane: {np.median(errors):.3f}s
        - Ecart-type: {np.std(errors):.3f}s
        - Min: {np.min(errors):.3f}s
        - Max: {np.max(errors):.3f}s
        
        Precision:
        - < 0.5s: {(errors < 0.5).sum() / len(errors) * 100:.1f}%
        - < 1.0s: {(errors < 1.0).sum() / len(errors) * 100:.1f}%
        - < 2.0s: {(errors < 2.0).sum() / len(errors) * 100:.1f}%
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                       family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nGraphiques sauvegardes: {save_path}")
        plt.show()
    
    def plot_training_history(self, save_path: str = "tensorflow_history.png"):
        """Visualise l'historique d'entraînement."""
        if self.history is None:
            print("Aucun historique d'entrainement disponible.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss - TensorFlow')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Metric
        metric_key = list(self.history.history.keys())[1]
        val_metric_key = 'val_' + metric_key
        axes[1].plot(self.history.history[metric_key], label=f'Train {metric_key}')
        axes[1].plot(self.history.history[val_metric_key], label=f'Val {metric_key}')
        axes[1].set_title(f'Model {metric_key} - TensorFlow')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_key)
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Historique d'entrainement sauvegarde: {save_path}")
        plt.show()


def main():
    """Fonction principale pour tester le modèle TensorFlow."""
    print("=" * 80)
    print("  MODELE TENSORFLOW - Analyse de Performances Automobiles")
    print("=" * 80)
    
    csv_path = Path("cars.csv")
    
    if not csv_path.exists():
        print("\nErreur: Fichier cars.csv introuvable.")
        return
    
    # Initialisation
    nn = TensorFlowCarNN()
    
    # Chargement
    df = nn.load_and_analyze(csv_path)
    
    # Préparation des données
    X_train, X_test, y_train, y_test = nn.prepare_regression_data(df)
    
    # Construction du modèle
    nn.calculate_optimal_params(len(X_train), X_train.shape[1])
    nn.build_regression_model(input_dim=X_train.shape[1])
    print("\nModele construit:")
    print(nn.model.summary())
    
    # Entraînement
    train_time = nn.train(X_train, y_train, X_test, y_test)
    
    # Évaluation
    mae, rmse, r2, predictions = nn.evaluate_regression(X_test, y_test)
    
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
    nn.plot_regression_results(y_test, predictions)
    nn.plot_training_history()
    
    print("\n" + "=" * 80)
    print("  ANALYSE TERMINEE")
    print("=" * 80)


if __name__ == "__main__":
    main()
