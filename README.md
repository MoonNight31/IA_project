# Project-IA : Analyse de Performances Automobiles avec Réseaux de Neurones

## 🎯 Objectif du Projet

Ce projet génère des datasets synthétiques de voitures avec des spécifications réalistes et utilise des réseaux de neurones pour analyser et prédire les performances automobiles.

## 📋 Fonctionnalités

### 1. Génération de Données
- **Simple Generator** (`program.py`) : Génération rapide avec propriétés de base (1000+ voitures)

### 2. Analyse par Réseaux de Neurones
- **Régression** : Prédiction du temps 0-100 km/h, vitesse max
- **Classification** : Prédiction du constructeur, type de carburant, etc.
- **Framework adaptatif** : Sélection automatique entre scikit-learn et TensorFlow selon la taille du dataset
- Support TensorFlow/Keras et scikit-learn

## 🚀 Installation

```bash
# Cloner le projet
git clone <votre-repo>
cd project-ia

# Installer les dépendances
pip install pandas matplotlib scikit-learn tensorflow

# Alternative avec Poetry
poetry install
```

## 💻 Utilisation

### Étape 1 : Générer les Données

```bash
# Génération simple (1000 voitures par défaut)
python src/project_ia/program.py
```

Cela crée un fichier `cars.csv` avec les données.

### Étape 2 : Entraîner les Réseaux de Neurones

#### Option 1 : Sélection Adaptative (Recommandé)
```bash
# Le script sélectionne automatiquement le meilleur framework
python src/project_ia/demo_adaptive.py
```

#### Option 2 : Framework Spécifique
```bash
# Utiliser scikit-learn (petits datasets < 50k)
python src/project_ia/model_sklearn.py

# Utiliser TensorFlow (grands datasets > 50k)
python src/project_ia/model_tensorflow.py
```

#### Option 3 : Utilisation depuis Python
```python
from pathlib import Path
from project_ia.model_sklearn import SkLearnCarNN
from project_ia.model_tensorflow import TensorFlowCarNN

# Avec scikit-learn
nn = SkLearnCarNN()
df = nn.load_and_analyze(Path("cars.csv"))
X_train, X_test, y_train, y_test = nn.prepare_regression_data(df)
train_time = nn.train(X_train, y_train, X_test, y_test)
mae, rmse, r2, predictions = nn.evaluate_regression(X_test, y_test)

# Avec TensorFlow
nn = TensorFlowCarNN()
df = nn.load_and_analyze(Path("cars.csv"))
X_train, X_test, y_train, y_test = nn.prepare_regression_data(df)
nn.build_regression_model(input_dim=X_train.shape[1])
train_time = nn.train(X_train, y_train, X_test, y_test)
mae, rmse, r2, predictions = nn.evaluate_regression(X_test, y_test)
```

## 📊 Structure des Données

### Fichier CSV généré (`cars.csv`)

| Colonne | Type | Description |
|---------|------|-------------|
| manufacturer | str | Constructeur (Renault, BMW, Ferrari, etc.) |
| model | str | Modèle de la voiture |
| year | int | Année de fabrication |
| power | int | Puissance (kW) |
| torque | int | Couple (Nm) |
| max_speed | int | Vitesse maximale (km/h) |
| fuel_efficiency | float | Consommation (L/100km ou kWh/100km) |
| fuel_type | str | Type de carburant (Gasoline, Diesel, Electric) |
| doors_number | int | Nombre de portes |
| weight | int | Poids (kg) |
| aerodynamic_level | float | Coefficient aérodynamique (Cx) |
| turbo_count | int | Nombre de turbos |
| millage_in_km | int | Kilométrage |
| zero_to_hundred | float | Temps 0-100 km/h (secondes) |
| transmission_type | str | Type de transmission (Manual, Automatic) |

## 🧠 Architectures des Réseaux de Neurones

### Architecture Adaptative
Le projet ajuste automatiquement l'architecture selon la taille du dataset :

**Petits datasets (< 50k échantillons)** - scikit-learn :
- MLPRegressor/MLPClassifier avec 2-3 couches cachées
- Batch size : 32-128
- Early stopping avec patience adaptative

**Grands datasets (> 50k échantillons)** - TensorFlow :
```
Input Layer (n features)
    ↓
Dense(72) + ReLU + Dropout(0.2)
    ↓
Dense(36) + ReLU + Dropout(0.2)
    ↓
Dense(18) + ReLU + Dropout(0.2)
    ↓
Output Layer (1 neuron pour régression, n classes pour classification)
```
- Batch size : 256-512
- Early stopping avec patience : 10-15 epochs

## 📈 Métriques de Performance

### Régression
- **MAE** (Mean Absolute Error) : Erreur moyenne absolue en secondes
- **RMSE** (Root Mean Squared Error) : Racine de l'erreur quadratique moyenne
- **R² Score** : Coefficient de détermination (0-1, plus proche de 1 = meilleur)

### Classification
- **Accuracy** : Précision globale
- **Precision/Recall/F1-Score** : Métriques par classe
- **Confusion Matrix** : Matrice de confusion

## 🎨 Visualisations

Le module génère automatiquement :
- **Prédictions vs Réalité** : Scatter plot des prédictions
- **Distribution des erreurs** : Histogramme des erreurs absolues
- **Erreurs par prédiction** : Analyse des erreurs
- **Statistiques détaillées** : Métriques et précision
- Graphiques sauvegardés en PNG (exemple : `adaptive_tensorflow_results.png`)

## 🔧 Configuration Avancée

### Sélection Automatique du Framework

Le système détecte automatiquement le meilleur framework :
- **< 50,000 échantillons** → scikit-learn (plus rapide)
- **> 50,000 échantillons** → TensorFlow (meilleure scalabilité)

### Paramètres Adaptatifs

Les hyperparamètres s'ajustent automatiquement :
- **Batch size** : 32 → 512 (selon taille dataset)
- **Architecture** : 2-3 couches (adapté au nombre de features)
- **Test size** : 20% → 10% (plus de données = moins de test nécessaire)
- **Epochs** : 200 → 50 (convergence plus rapide avec plus de données)

### Changer la Cible de Prédiction

```python
# Prédire la vitesse max au lieu du 0-100
X_train, X_test, y_train, y_test = nn.prepare_regression_data(
    df, target_col="max_speed"
)

# Classifier par constructeur
X_train, X_test, y_train, y_test = nn.prepare_classification_data(
    df, target_col="manufacturer"
)
```

## 📁 Structure du Projet

```
project-ia/
├── src/
│   └── project_ia/
│       ├── __init__.py
│       ├── program.py           # Générateur simple de voitures
│       ├── model_sklearn.py     # Modèle scikit-learn
│       ├── model_tensorflow.py  # Modèle TensorFlow/Keras
│       └── demo_adaptive.py     # Sélection automatique framework
├── tests/
│   └── __init__.py
├── .github/
│   └── copilot-instructions.md  # Guide pour IA
├── cars.csv                      # Dataset généré (198k voitures)
├── pyproject.toml               # Configuration Poetry
└── README.md                    # Ce fichier
```

## 🎓 Exemples de Résultats

### Dataset de 198,000 voitures
- **Framework sélectionné** : TensorFlow (automatique)
- **R² Score** : 99.4% (excellente précision)
- **MAE** : ~0.14 secondes
- **Temps d'entraînement** : ~20 secondes
- **Vitesse** : 8,786 échantillons/seconde

### Cas d'Usage

**1. Prédiction de Performance (0-100 km/h)**
- Entrée : puissance, poids, aérodynamique, turbos
- Sortie : temps 0-100 km/h avec précision < 0.5s dans 95% des cas

**2. Classification de Constructeur**
- Entrée : caractéristiques techniques
- Sortie : constructeur prédit (Renault, BMW, Ferrari, etc.)

**3. Analyse de Vitesse Maximale**
- Entrée : puissance, Cx, poids
- Sortie : vitesse max prédite

## 🐛 Dépannage

### Erreur d'importation de modules
```bash
# Le projet ajuste automatiquement le PYTHONPATH
# Exécutez toujours depuis la racine du projet
cd project-ia
python src/project_ia/demo_adaptive.py
```

### TensorFlow non disponible
Si TensorFlow n'est pas installé, le système bascule automatiquement sur scikit-learn.
```bash
pip install tensorflow
```

### Fichier cars.csv introuvable
Le projet inclut déjà un dataset. Si besoin d'en générer un nouveau :
```bash
python src/project_ia/program.py
```

### Performance lente
- Petits datasets : Le système utilise automatiquement scikit-learn (plus rapide)
- Grands datasets : TensorFlow est plus efficace mais nécessite plus de setup time

## 📝 Licence

MoonNight31


