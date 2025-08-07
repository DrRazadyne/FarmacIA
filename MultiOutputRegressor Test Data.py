import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
fechas = pd.date_range(start="2023-01-01", end="2024-12-31", freq="ME")
n_meses = len(fechas)

atb = ["Ceftriaxona", "Metronidazol", "PTZ", "Vancomicina", "AMS"]
df = pd.DataFrame({
    "Fecha": fechas,
    "Cirugías": np.random.poisson(lam=30, size=n_meses),
    "Mes": fechas.month
})

df["Estación"] = df["Mes"].apply(lambda m: "Verano" if m in [12, 1, 2] else "Invierno" if m in [6, 7, 8] else "Otro")
est_dummies = pd.get_dummies(df["Estación"], drop_first=True)
X_base = pd.concat([df[["Cirugías"]], est_dummies], axis=1)

for a in atb:
    df[a] = np.random.normal(loc=100 + 5 * df["Cirugías"], scale=10, size=n_meses)

modelos = {
    "LinearRegression": LinearRegression(),
    "BayesianRidge": BayesianRidge(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "KNeighbors": KNeighborsRegressor(),
    "SVR": SVR(),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(50,), max_iter=10000, random_state=42)
}

resultados = []
for a in atb:
    y = df[a]
    for nombre, modelo in modelos.items():
        X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)

        resultados.append({
            "Antibiótico": a,
            "Modelo": nombre,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "R²": round(r2, 2)
        })

df_resultados = pd.DataFrame(resultados)
print("\nMétricas por modelo y antibiótico:")
print(df_resultados)

plt.figure(figsize=(14, 6))
sns.barplot(data=df_resultados, x="Modelo", y="R²", hue="Antibiótico", palette="viridis")
plt.title("Comparación de R² por modelo y antibiótico")
plt.ylabel("R²")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
