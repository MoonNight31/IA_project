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
import warnings
warnings.filterwarnings('ignore')

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
            # Configurer TensorFlow pour éviter les erreurs GPU/mémoire
            tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None
            self.tf = tf
            self.keras = keras
            print(f"TensorFlow version: {tf.__version__}")
        except ImportError:
            raise ImportError("TensorFlow n'est pas installé. Exécutez: python -m pip install tensorflow")
        except Exception as e:
            print(f"Avertissement TensorFlow: {e}")
            import tensorflow as tf
            from tensorflow import keras
            self.tf = tf
            self.keras = keras
        
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
        
        try:
            # Spécifier l'encodage pour éviter les erreurs
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='latin-1')
            except:
                df = pd.read_csv(csv_path, encoding='cp1252')
        
        # Nettoyer les données
        df = df.dropna()
        if 'is_started' in df.columns:
            df = df.drop('is_started', axis=1)
        
        # Vérifier les colonnes nécessaires
        required_cols = ['power', 'weight', 'zero_to_hundred']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        load_time = time.time() - start_time
        
        print(f"  Dataset chargé en {load_time:.2f}s")
        print(f"  Taille: {len(df):,} voitures")
        print(f"  Colonnes: {list(df.columns)}")
        print(f"  Mémoire: ~{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
    
    def prepare_regression_data(self, df: pd.DataFrame, target_col: str = "zero_to_hundred"):
        """Prépare les données pour la régression."""
        print("\nPréparation des données pour la régression...")
        
        # Vérifier que la colonne cible existe
        if target_col not in df.columns:
            raise ValueError(f"Colonne cible '{target_col}' introuvable dans le dataset")
        
        # Features numériques - vérifier leur existence
        available_numeric = ['power', 'torque', 'weight', 'aerodynamic_level', 
                           'turbo_count', 'year', 'doors_number']
        numeric_features = [col for col in available_numeric if col in df.columns]
        
        # Features catégorielles - vérifier leur existence
        available_categorical = ['fuel_type', 'transmission_type', 'manufacturer']
        categorical_features = [col for col in available_categorical if col in df.columns]
        
        print(f"  Features numériques: {numeric_features}")
        print(f"  Features catégorielles: {categorical_features}")
        
        df_processed = df.copy()
        
        # One-hot encoding avec gestion d'erreurs
        encoded_features = []
        for cat_feat in categorical_features:
            try:
                # Nettoyer les valeurs nulles et les espaces
                df_processed[cat_feat] = df_processed[cat_feat].astype(str).str.strip()
                dummies = pd.get_dummies(df_processed[cat_feat], prefix=cat_feat, drop_first=True)
                encoded_features.extend(dummies.columns.tolist())
                df_processed = pd.concat([df_processed, dummies], axis=1)
            except Exception as e:
                print(f"  Erreur avec {cat_feat}: {e}")
                continue
        
        feature_cols = numeric_features + encoded_features
        
        # Vérifier qu'il y a assez de features
        if len(feature_cols) < 3:
            raise ValueError(f"Pas assez de features valides: {feature_cols}")
        
        # Extraire X et y avec gestion d'erreurs
        try:
            X = df_processed[feature_cols].values.astype(np.float32)
            y = df[target_col].values.astype(np.float32)
        except Exception as e:
            print(f"Erreur lors de l'extraction des données: {e}")
            # Essayer de nettoyer les données
            for col in feature_cols:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed = df_processed.dropna()
            X = df_processed[feature_cols].values.astype(np.float32)
            y = df_processed[target_col].values.astype(np.float32)
        
        # Vérifier les valeurs infinies ou NaN
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("  Nettoyage des valeurs NaN/Inf...")
            finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[finite_mask]
            y = y[finite_mask]
        
        self.feature_names = feature_cols
        n_features = len(feature_cols)
        
        # Calculer les paramètres optimaux
        config = self.calculate_optimal_params(len(X), n_features)
        
        print(f"\nConfiguration TensorFlow:")
        print(f"  Features: {n_features}")
        print(f"  Échantillons: {len(X):,}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Max epochs: {config['max_epochs']}")
        print(f"  Patience: {config['patience']}")
        print(f"  Hidden layers: {config['hidden_layers']}")
        print(f"  Test size: {config['test_size']*100:.0f}%")
        
        # Split avec vérification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config['test_size'], random_state=42
            )
        except Exception as e:
            print(f"Erreur lors du split: {e}")
            # Fallback avec test_size plus petit
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Normalisation avec gestion d'erreurs
        print("\nNormalisation des données...")
        try:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        except Exception as e:
            print(f"Erreur lors de la normalisation: {e}")
            # Utiliser une normalisation simple
            X_mean = np.mean(X_train, axis=0)
            X_std = np.std(X_train, axis=0)
            X_std[X_std == 0] = 1  # Éviter la division par zéro
            X_train = (X_train - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std
        
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
        try:
            model = self.keras.Sequential()
            
            # Première couche avec gestion d'erreur
            model.add(self.keras.layers.Dense(
                self.config['hidden_layers'][0], 
                activation='relu', 
                input_dim=input_dim,
                kernel_initializer='he_normal'
            ))
            model.add(self.keras.layers.Dropout(0.2))
            
            # Couches cachées
            for units in self.config['hidden_layers'][1:]:
                model.add(self.keras.layers.Dense(
                    units, 
                    activation='relu',
                    kernel_initializer='he_normal'
                ))
                model.add(self.keras.layers.Dropout(0.2))
            
            # Couche de sortie
            model.add(self.keras.layers.Dense(1, kernel_initializer='he_normal'))
            
            # Compilation avec learning rate adaptatif
            optimizer = self.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            self.model = model
            return model
            
        except Exception as e:
            print(f"Erreur lors de la construction du modèle: {e}")
            raise
    
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
        print("ENTRAÎNEMENT - TensorFlow/Keras")
        print("=" * 80)
        print(f"Architecture: {self.config['hidden_layers']}")
        
        try:
            # Callbacks avec gestion d'erreur
            callbacks = []
            
            # Early stopping
            early_stopping = self.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=self.config['patience'], 
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Reduce learning rate
            reduce_lr = self.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['patience']//2,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            # Entraînement
            start_time = time.time()
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.config['max_epochs'],
                batch_size=self.config['batch_size'],
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            train_time = time.time() - start_time
            
            print(f"\nEntraînement terminé en {train_time:.2f}s")
            print(f"Epochs exécutés: {len(self.history.history['loss'])}")
            
            return train_time
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement: {e}")
            raise
    
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
        try:
            # Configurer matplotlib pour éviter les erreurs d'affichage
            plt.style.use('default')
            
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
            
            # Sauvegarder avec gestion d'erreur
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"\nGraphiques sauvegardés: {save_path}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde: {e}")
            
            # Afficher seulement si possible
            try:
                plt.show()
            except:
                print("Impossible d'afficher les graphiques (mode headless)")
                plt.close()
                
        except Exception as e:
            print(f"Erreur lors de la création des graphiques: {e}")
    
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
    print("  MODÈLE TENSORFLOW - Analyse de Performances Automobiles")
    print("=" * 80)
    
    csv_path = Path("cars.csv")
    
    if not csv_path.exists():
        print(f"\nErreur: Fichier {csv_path} introuvable.")
        print("Exécutez d'abord program.py pour générer le dataset.")
        return
    
    try:
        # Initialisation
        nn = TensorFlowCarNN()
        
        # Chargement
        df = nn.load_and_analyze(csv_path)
        
        # Préparation des données
        X_train, X_test, y_train, y_test = nn.prepare_regression_data(df)
        
        # Construction du modèle
        nn.build_regression_model(input_dim=X_train.shape[1])
        print(f"\nModèle construit avec {X_train.shape[1]} features d'entrée")
        
        # Entraînement
        train_time = nn.train(X_train, y_train, X_test, y_test)
        
        # Évaluation
        mae, rmse, r2, predictions = nn.evaluate_regression(X_test, y_test)
        
        # Résultats
        print("\n" + "=" * 80)
        print("  RÉSULTATS FINAUX")
        print("=" * 80)
        print(f"\nMétriques de performance:")
        print(f"  MAE  : {mae:.3f} secondes")
        print(f"  RMSE : {rmse:.3f} secondes")
        print(f"  R²   : {r2:.3f} ({r2*100:.1f}%)")
        
        print(f"\nPerformances d'exécution:")
        print(f"  Temps d'entraînement: {train_time:.2f}s")
        print(f"  Vitesse: {len(X_train) / train_time:.0f} échantillons/s")
        
        # Visualisation
        print(f"\nGénération des graphiques...")
        nn.plot_regression_results(y_test, predictions)
        nn.plot_training_history()
        
        print("\n" + "=" * 80)
        print("  ANALYSE TERMINÉE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nErreur critique: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
