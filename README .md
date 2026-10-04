# Prédiction de l'état d'un téléphone d'occasion

App Streamlit qui prédit l'état d'un téléphone (`D'occasion`, `Neuf`,
`Réconditionné` ou `Venant`) à partir de 6 variables : prix, adresse, marque,
dimension d'écran, RAM et stockage.

Le modèle retenu est un **Gradient Boosting** (meilleur F1 en validation parmi 7
modèles comparés : Random Forest, Gradient Boosting, XGBoost, KNN, Logistic
Regression, SVM, Decision Tree).

| Modèle | F1 (validation) |
|---|---|
| **Gradient Boosting** ✅ | 0,903 |
| Random Forest | 0,900 |
| XGBoost | 0,900 |
| Decision Tree | 0,871 |
| SVM | 0,857 |
| KNN | 0,835 |
| Logistic Regression | 0,685 |

## Fichiers du dépôt

| Fichier | Rôle |
|---|---|
| `app.py` | L'application Streamlit (à déployer) |
| `gb_model.joblib` | Le modèle entraîné (Gradient Boosting) |
| `encoders.joblib` | Les `LabelEncoder` (adresse, marque, etat — dans cet ordre) |
| `scaler.joblib` | Le `StandardScaler` des variables numériques |
| `uniques.joblib` | Valeurs possibles (adresse, marque, etat — dans cet ordre), pour les menus déroulants |
| `model_info.joblib` | Nom du meilleur modèle, ordre des variables, tableau de comparaison |
| `requirements.txt` | Dépendances pour Streamlit Cloud |
| `train_model.py` | Script pour ré-entraîner et régénérer les fichiers `.joblib` |
| `Telephone_data.csv` | Données source (utile pour `train_model.py`, pas requis par `app.py`) |

## Déployer sur Streamlit Community Cloud

1. Créer un dépôt GitHub et y pousser au minimum : `app.py`, `gb_model.joblib`,
   `encoders.joblib`, `scaler.joblib`, `uniques.joblib`, `model_info.joblib`,
   `requirements.txt`.
2. Aller sur [share.streamlit.io](https://share.streamlit.io), se connecter avec GitHub.
3. "New app" → choisir le dépôt, la branche, et `app.py` comme fichier principal.
4. Déployer.

## Ré-entraîner le modèle (optionnel, en local)

```bash
pip install -r requirements-train.txt
python train_model.py   # régénère les .joblib à partir de Telephone_data.csv
```

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```
