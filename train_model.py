"""
Reproduit le pipeline du TP3 (classification de l'état d'un téléphone d'occasion :
D'occasion / Neuf / Reconditionné / Venant) et sauvegarde les éléments nécessaires
au déploiement dans gb_model.joblib / encoders.joblib / scaler.joblib / uniques.joblib.

Note : les grilles de recherche d'hyperparamètres (GridSearchCV) ont été resserrées
par rapport au notebook d'origine pour que l'entrainement complet (7 modèles) prenne
quelques minutes au lieu d'une à deux heures. Les mêmes familles de modèles sont
comparées et le classement obtenu est le même que dans le notebook (Gradient
Boosting en tête). Le SVM n'utilise pas `probability=True` (inutile pour ce
déploiement et ça multiplie le temps d'entrainement par 5 à 10x).

À lancer une seule fois (en local), AVANT de déployer :
    pip install -r requirements-train.txt
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib as jb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

DATA_PATH = "Telephone_data.csv"

# 1. Charger les données
df = pd.read_csv(DATA_PATH)

cat_cols = df.select_dtypes(include="object").columns.tolist()
quant_cols = df.select_dtypes(include="number").columns.tolist()

# 2. Sauvegarder les valeurs uniques des variables catégorielles (pour les menus déroulants de l'app)
uniques = [list(np.unique(df[col])) for col in cat_cols]
jb.dump(uniques, "uniques.joblib")

# 3. Encoder les variables catégorielles (adresse, marque, etat)
encoders = []
for col in cat_cols:
    enc = LabelEncoder()
    enc.fit(df[col])
    df[col] = enc.transform(df[col])
    encoders.append(enc)
jb.dump(encoders, "encoders.joblib")

# 4. Imputation des outliers (méthode IQR + KNNImputer), comme dans le notebook
def impute_func(df, col, n_neighbors=3):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    min_value = Q1 - 1.5 * IQR
    max_value = Q3 + 1.5 * IQR
    df.loc[(df[col] < min_value) | (df[col] > max_value), col] = np.nan
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[:] = imputer.fit_transform(df)

for col in quant_cols:
    impute_func(df, col)

# Supprimer les outliers restants sur 'ram' (comme dans le notebook)
Q1 = df["ram"].quantile(0.25)
Q3 = df["ram"].quantile(0.75)
IQR = Q3 - Q1
min_value, max_value = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
indexes = list(df.loc[(df["ram"] < min_value) | (df["ram"] > max_value), "ram"].index)
df.drop(indexes, axis=0, inplace=True)

# 5. Fractionner puis normaliser
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
feature_cols = df.columns[:-1].tolist()

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
jb.dump(scaler, "scaler.joblib")

x_train, x_vt, y_train, y_vt = train_test_split(x, y, train_size=0.8, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_vt, y_vt, train_size=0.5, random_state=42)


def evaluation(model, x_eval=x_val, y_eval=y_val):
    y_pred = model.predict(x_eval)
    acc = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred, average="micro")
    prec = precision_score(y_eval, y_pred, average="micro")
    rec = recall_score(y_eval, y_pred, average="micro")
    return acc, f1, prec, rec


resultats = {}
modeles = {}

# --- Random Forest ---
params_rf = {"n_estimators": [10, 30, 50, 70, 90], "max_depth": [3, 6, 9, 12, None], "criterion": ["gini", "entropy"]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), params_rf, cv=5, n_jobs=-1)
grid_rf.fit(x_train, y_train)
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1, **grid_rf.best_params_)
rf_model.fit(x_train, y_train)
modeles["Random Forest"] = rf_model
resultats["Random Forest"] = evaluation(rf_model)

# --- Gradient Boosting ---
params_gb = {"n_estimators": [10, 30, 50, 70, 90], "max_depth": [3, 6, 9, 12]}
grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), params_gb, cv=5, n_jobs=-1)
grid_gb.fit(x_train, y_train)
gb_model = GradientBoostingClassifier(random_state=42, **grid_gb.best_params_)
gb_model.fit(x_train, y_train)
modeles["Gradient Boosting"] = gb_model
resultats["Gradient Boosting"] = evaluation(gb_model)

# --- XGBoost ---
params_xgb = {"n_estimators": [10, 30, 50, 70, 90], "max_depth": [3, 6, 9, 12]}
grid_xgb = GridSearchCV(XGBClassifier(random_state=42, n_jobs=-1), params_xgb, cv=5, n_jobs=-1)
grid_xgb.fit(x_train, y_train)
xgb_model = XGBClassifier(random_state=42, n_jobs=-1, **grid_xgb.best_params_)
xgb_model.fit(x_train, y_train)
modeles["XGBoost"] = xgb_model
resultats["XGBoost"] = evaluation(xgb_model)

# --- KNN ---
params_knn = {"n_neighbors": list(np.arange(2, 31))}
grid_knn = GridSearchCV(KNeighborsClassifier(n_jobs=-1), params_knn, cv=5, n_jobs=-1)
grid_knn.fit(x_train, y_train)
knn_model = KNeighborsClassifier(n_jobs=-1, **grid_knn.best_params_)
knn_model.fit(x_train, y_train)
modeles["KNN"] = knn_model
resultats["KNN"] = evaluation(knn_model)

# --- Logistic Regression ---
params_lg = {"C": np.logspace(-3, 3, 7)}
grid_lg = GridSearchCV(LogisticRegression(max_iter=1000), params_lg, cv=5, n_jobs=-1)
grid_lg.fit(x_train, y_train)
lg_model = LogisticRegression(max_iter=1000, **grid_lg.best_params_)
lg_model.fit(x_train, y_train)
modeles["Logistic Regression"] = lg_model
resultats["Logistic Regression"] = evaluation(lg_model)

# --- SVM ---
params_svm = {"C": np.logspace(-1, 3, 5), "kernel": ["linear", "rbf"]}
grid_svm = GridSearchCV(SVC(random_state=42), params_svm, cv=5, n_jobs=-1)
grid_svm.fit(x_train, y_train)
svm_model = SVC(random_state=42, **grid_svm.best_params_)
svm_model.fit(x_train, y_train)
modeles["SVM"] = svm_model
resultats["SVM"] = evaluation(svm_model)

# --- Decision Tree ---
params_dt = {"criterion": ["gini", "entropy"], "max_depth": list(np.arange(3, 15))}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), params_dt, cv=5, n_jobs=-1)
grid_dt.fit(x_train, y_train)
dt_model = DecisionTreeClassifier(random_state=42, **grid_dt.best_params_)
dt_model.fit(x_train, y_train)
modeles["Decision Tree"] = dt_model
resultats["Decision Tree"] = evaluation(dt_model)

# 6. Comparaison et sélection du meilleur modèle (par F1 score de validation)
results_df = pd.DataFrame(
    [{"Modèle": nom, "Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec}
     for nom, (acc, f1, prec, rec) in resultats.items()]
).sort_values(by="F1", ascending=False).reset_index(drop=True)
print(results_df)

best_model_name = results_df.loc[0, "Modèle"]
best_model = modeles[best_model_name]
print(f"\nModèle le plus performant : {best_model_name} (F1 val = {results_df.loc[0, 'F1']:.4f})")

# 7. Évaluation finale sur le jeu de test
y_pred_test = best_model.predict(x_test)
print(f"Accuracy test : {accuracy_score(y_test, y_pred_test):.4f}")
print(f"F1 test       : {f1_score(y_test, y_pred_test, average='micro'):.4f}")

# 8. Sauvegarder le modèle et les infos nécessaires au déploiement
jb.dump(best_model, "gb_model.joblib")  # nom gardé par cohérence avec le notebook, même si un autre modèle gagne
jb.dump({"best_model_name": best_model_name, "feature_cols": feature_cols, "results": results_df}, "model_info.joblib")
print("\nSauvegardé : gb_model.joblib, encoders.joblib, scaler.joblib, uniques.joblib, model_info.joblib")
