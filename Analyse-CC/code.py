# ==============================================================================
# 1. Importations et gestion des avertissements
# ==============================================================================
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For ignoring warning
import warnings
warnings.filterwarnings("ignore")

# NOTE: Les importations de scikit-learn sont implicitement requises pour que les blocs ci-dessous fonctionnent.
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Il est supposé que X_train, y_train, X_test, y_test, results, 
# y_pred_lin_reg et y_pred_poly_reg sont définis dans les étapes de pré-traitement précédentes.


# ==============================================================================
# 2. Chargement des données et forme (df.shape omis)
# ==============================================================================
df=pd.read_csv('/content/drive/MyDrive/preparation examen/03 dec 2025 /credit_card_fraud_dataset.csv')
# df 


# ==============================================================================
# 3. Corrélation et Visualisation (Heatmap)
# ==============================================================================
# Calcul de la matrice de corrélation
correlation_matrix = df.corr()

# Visualisation de la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(18, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de Corrélation des Caractéristiques')
plt.show()


# ==============================================================================
# 4. Régression par arbre de décision (Decision Tree Regression)
# ==============================================================================
model_name = 'Decision Tree Regression'
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train) 
y_pred_dt_reg = dt_reg.predict(X_test)

mae_dt_reg = mean_absolute_error(y_test, y_pred_dt_reg) 
mse_dt_reg = mean_squared_error(y_test, y_pred_dt_reg) 
rmse_dt_reg = np.sqrt(mse_dt_reg)
r2_dt_reg = r2_score(y_test, y_pred_dt_reg) 

results[model_name] = { 
 'MAE': mae_dt_reg,
 'MSE': mse_dt_reg,
 'RMSE': rmse_dt_reg,
 'R-squared': r2_dt_reg
}


# ==============================================================================
# 5. Régression par forêt aléatoire (Random Forest Regression)
# ==============================================================================
model_name = 'Random Forest Regression'
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf_reg = rf_reg.predict(X_test)

mae_rf_reg = mean_absolute_error(y_test, y_pred_rf_reg)
mse_rf_reg = mean_squared_error(y_test, y_pred_rf_reg)
rmse_rf_reg = np.sqrt(mse_rf_reg)
r2_rf_reg = r2_score(y_test, y_pred_rf_reg)

results[model_name] = {
 'MAE': mae_rf_reg,
 'MSE': mse_rf_reg,
 'RMSE': rmse_rf_reg,
 'R-squared': r2_rf_reg
}


# ==============================================================================
# 6. Régression par Support Vector Regression (SVR)
# (Inclut Scaling et Sampling pour performance)
# ==============================================================================
model_name = 'SVR'

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Due to SVR's computational cost, sample a smaller subset for training
sample_size = 10000
if len(X_train_scaled) > sample_size:
 np.random.seed(42) 
 sample_indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
 X_train_svr_sampled = X_train_scaled[sample_indices]
 y_train_svr_sampled = y_train.iloc[sample_indices]
else:
 X_train_svr_sampled = X_train_scaled
 y_train_svr_sampled = y_train

svr_reg = SVR(kernel='rbf') 
svr_reg.fit(X_train_svr_sampled, y_train_svr_sampled)
y_pred_svr = svr_reg.predict(X_test_scaled)

mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_pred_svr)

results[model_name] = {
 'MAE': mae_svr,
 'MSE': mse_svr,
 'RMSE': rmse_svr,
 'R-squared': r2_svr
}


# ==============================================================================
# 7. Régression par Gradient Boosting (Gradient Boosting Regression)
# ==============================================================================
model_name = 'Gradient Boosting Regression'
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)

mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

results[model_name] = {
 'MAE': mae_gbr,
 'MSE': mse_gbr,
 'RMSE': rmse_gbr,
 'R-squared': r2_gbr
}


# ==============================================================================
# 8. Affichage et Visualisation des résultats
# ==============================================================================
results_df = pd.DataFrame(results).T
# print("Performance metrics for all regression models:")
# print(results_df.round(3))

model_predictions = [ 
 ('Linear Regression', y_pred_lin_reg), 
 ('Polynomial Regression (Degree 2)', y_pred_poly_reg), 
 ('Decision Tree Regression', y_pred_dt_reg), 
 ('Random Forest Regression', y_pred_rf_reg), 
 ('SVR', y_pred_svr), 
 ('Gradient Boosting Regression', y_pred_gbr) 
]

for name, y_pred in model_predictions: 
 fig, axes = plt.subplots(1, 2, figsize=(16, 6))

 # Plot Actual vs. Predicted values
 axes[0].scatter(y_test, y_pred, alpha=0.6)
 axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
 axes[0].set_xlabel('Valeurs réelles')
 axes[0].set_ylabel('Valeurs prédites')
 axes[0].set_title(f'Valeurs réelles vs. prédites pour {name}')

 # Calculate residuals
 residuals = y_test - y_pred

 # Plot Residuals
 axes[1].scatter(y_pred, residuals, alpha=0.6)
 axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
 axes[1].set_xlabel('Valeurs prédites')
 axes[1].set_ylabel('Résidus')
 axes[1].set_title(f'Diagramme des résidus pour {name}')

 plt.tight_layout()
 plt.show()
