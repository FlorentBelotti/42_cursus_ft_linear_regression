# TEST.md — Guide de correction

Procédure complète pour démontrer chaque point de la grille de correction, dans l'ordre.

---

## Préliminaires

### Il existe deux programmes distincts
Montrer les deux fichiers à la racine du dépôt :

```
predict_car_price.py          ← programme de prédiction
train_linear_regression_model.py  ← programme d'apprentissage
```

### Absence de bibliothèque interdite
Montrer les imports des deux entrées et des modules métier.
Aucun `numpy`, `scipy`, `sklearn` ou équivalent n'apparaît.
Les seules bibliothèques utilisées sont `json` et `os` (bibliothèque standard Python).

---

## Point 1 — Prédiction avant apprentissage

**Ce qu'on vérifie :** le programme retourne 0 pour tout kilométrage quand aucun apprentissage n'a encore eu lieu.

**Comment le montrer :**

```bash
# Supprimer les paramètres sauvegardés si nécessaire
rm -f model_parameters.json

# Lancer le prédicteur et entrer un kilométrage non nul (ex: 100000)
python3 predict_car_price.py
```

**Résultat attendu :**
```
Enter mileage (km): 100000
Estimated price: 0.00
```

**Pourquoi c'est 0 :** sans fichier `model_parameters.json`, `ModelParameterStorage.load()`
retourne `(0.0, 0.0)`. L'équation `θ0 + θ1 × mileage` donne donc `0 + 0 × 100000 = 0`.

**Équation à montrer dans le code :**

```python
# model/linear_regression_model.py
def predict(self, mileage: float) -> float:
    return self._theta0 + self._theta1 * mileage   # θ0 + (θ1 × x)
```

---

## Point 2 — Phase d'apprentissage

**Ce qu'on vérifie :** l'équation du sujet est implémentée, et θ0/θ1 sont sauvegardés à la fin.

**Comment le montrer :**

```bash
python3 train_linear_regression_model.py
```

**Résultat attendu :**
```
Training complete (1000 iterations, lr=0.1)
θ0 = 8481.172797
θ1 = -0.0212737191
```

**Équations à montrer dans le code** (`training/gradient_descent_trainer.py`) :

```python
errors = [model.predict(mileages[i]) - prices[i] for i in range(m)]
tmp_theta0 = self._learning_rate * sum(errors) / m
tmp_theta1 = self._learning_rate * sum(errors[i] * mileages[i] for i in range(m)) / m
model.update_parameters(
    model.theta0 - tmp_theta0,
    model.theta1 - tmp_theta1,
)
```

Ce sont exactement les formules du sujet :
```
tmpθ0 = learningRate × (1/m) × Σ (estimatePrice(mileage[i]) − price[i])
tmpθ1 = learningRate × (1/m) × Σ (estimatePrice(mileage[i]) − price[i]) × mileage[i]
```

**Vérifier la sauvegarde :**

```bash
cat model_parameters.json
```

```json
{"theta0": 8481.172796984529, "theta1": -0.021273719070404658}
```

θ0 et θ1 sont bien persistés dans `model_parameters.json` à l'issue de l'entraînement.

---

## Point 3 — Lecture du fichier CSV

**Ce qu'on vérifie :** le programme d'apprentissage lit bien le CSV comme base de données.

**Montrer le code** (`data_loading/csv_dataset_loader.py`) :

```python
def load(self) -> tuple[list[float], list[float]]:
    with open(self._filepath, newline="") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:          # skip header
        parts = line.split(",")
        mileages.append(float(parts[0].strip()))
        prices.append(float(parts[1].strip()))
    return mileages, prices
```

**Montrer l'appel dans le script d'entraînement** (`train_linear_regression_model.py`) :

```python
loader = CsvDatasetLoader(dataset_file)   # dataset_file → "data.csv"
mileages, prices = loader.load()
```

Le fichier `data.csv` est lu, la ligne d'en-tête (`km,price`) est ignorée,
et les 24 paires kilométrage/prix sont chargées en mémoire pour l'entraînement.

---

## Point 4 — Assignation simultanée de theta0 et theta1

**Ce qu'on vérifie :** les deux paramètres sont calculés dans des variables temporaires
avant d'être assignés ensemble, pas l'un après l'autre.

**Montrer le code** (`training/gradient_descent_trainer.py`) :

```python
# 1. Calcul des erreurs avec les theta ACTUELS
errors = [model.predict(mileages[i]) - prices[i] for i in range(m)]

# 2. Stockage des gradients dans des variables TEMPORAIRES
tmp_theta0 = self._learning_rate * sum(errors) / m
tmp_theta1 = self._learning_rate * sum(errors[i] * mileages[i] for i in range(m)) / m

# 3. Mise à jour SIMULTANEE : les deux assignations utilisent les anciens theta
model.update_parameters(
    model.theta0 - tmp_theta0,
    model.theta1 - tmp_theta1,
)
```

**Pourquoi c'est important :** si θ0 était mis à jour en premier, le calcul de θ1 utiliserait
déjà le nouveau θ0. Les variables temporaires garantissent que les deux gradients sont
calculés à partir du même état, puis appliqués simultanément.

---

## Point 5 — Prédiction après apprentissage

**Ce qu'on vérifie :** le prédicteur donne un prix cohérent avec le CSV, sans être exact.

**Kilométrages recommandés à tester** (valeurs présentes dans `data.csv`) :

```bash
python3 predict_car_price.py   # entrer : 240000
python3 predict_car_price.py   # entrer : 82029
python3 predict_car_price.py   # entrer : 22899
```

**Résultats obtenus vs CSV :**

| Kilométrage | Prix prédit | Prix réel (CSV) | Écart |
|-------------|------------|-----------------|-------|
| 240 000 km  | 3 375 €    | 3 650 €         | ~7.5 % |
| 82 029 km   | 6 736 €    | 6 390 €         | ~5.4 % |
| 22 899 km   | 7 994 €    | 7 990 €         | ~0.05 % |

Les prix sont cohérents : plus le kilométrage est élevé, plus le prix est bas.
Un écart existe — c'est normal et attendu dans une régression linéaire (le modèle
lisse la tendance générale, il ne mémorise pas chaque point).

**Note sur l'overfitting** (point bonus si demandé) :
L'overfitting désigne un modèle qui a trop « appris » ses données d'entraînement au
point de les reproduire à la valeur exacte. Un tel modèle prédit parfaitement les données
connues, mais échoue sur toute nouvelle valeur. Ici, les écarts observés (~5-8 %) montrent
que le modèle a bien generalisé la tendance sans sur-apprendre.
