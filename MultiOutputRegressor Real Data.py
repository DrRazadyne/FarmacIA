import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import re

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
atb_list = ["Ceftriaxona", "Metronidazol", "PTZ", "Vancomicina", "AMS"]

def parse_date_string(s):
    text = str(s)
    m = re.search(r"(\d{1,2}[\/\-.]\d{1,2}(?:[\/\-.]\d{2,4})?)", text)
    if not m:
        return None
    part = m.group(1)
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d-%m-%y", "%d.%m.%y"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(part, fmt)
            if dt.year < 100:
                yy = dt.year
                if 23 <= yy <= 25:
                    dt = dt.replace(year=2000 + yy)
                else:
                    return None
            return dt
        except ValueError:
            continue
    return None

cirugias_file = r"D:\CIRA\FarmacIA\Cirugias por año y por mes 2023 a 2025.xlsx"
df = pd.read_excel(cirugias_file)

df["Total"] = df["Planta"] + df["Guardia"]

df["Fecha"] = pd.to_datetime(
    df["Año"].astype(int).astype(str) + "-" +
    df["Mes"].astype(int).astype(str) + "-01"
)

df["Estación"] = df["Mes"].map(
    lambda m: "Verano" if m in [12,1,2]
              else "Invierno" if m in [6,7,8]
              else "Otro"
)
est_dummies = pd.get_dummies(df["Estación"], drop_first=True)

X_base = pd.concat([df[["Total"]], est_dummies], axis=1)

salida_file = r"D:\CIRA\FarmacIA\De sala a farmacia - ATB (_entradas_) del 01-01-23 al 30-06-25.xlsx"
xls_sal = pd.ExcelFile(salida_file)

for atb in atb_list:
    df_ent = xls_sal.parse(atb, header=None)
    df_ent = xls_sal.parse(atb, header=None, usecols=[0, 1])
    df_ent.columns = ["raw_fecha", "Cantidad"]
    df_ent["Fecha_parsed"] = df_ent["raw_fecha"].apply(parse_date_string)
    df_ent = df_ent.dropna(subset=["Fecha_parsed"])
    df_ent["Year"]  = df_ent["Fecha_parsed"].dt.year
    df_ent["Month"] = df_ent["Fecha_parsed"].dt.month
    monthly = (
        df_ent.groupby(["Year", "Month"])["Cantidad"]
        .sum()
        .reset_index()
        .rename(columns={
            "Cantidad": f"{atb}_Entregado"
        })
    )
    df = df.merge(
        monthly,
        left_on=["Año", "Mes"],
        right_on=["Year", "Month"],
        how="left"
    )
    df[f"{atb}_Entregado"] = df[f"{atb}_Entregado"].fillna(0)
    df.drop(columns=["Year", "Month"], inplace=True)

entrada_file = r"D:\CIRA\FarmacIA\De farmacia a sala - ATB (_salidas_) del 01-01-23 al 30-06-25.xlsx"
xls_dev = pd.ExcelFile(entrada_file)

for atb in atb_list:
    df_dev = xls_dev.parse(atb, header=None)
    df_dev.columns = ["raw_fecha", "Cantidad"]
    df_dev["Fecha_parsed"] = df_dev["raw_fecha"].apply(parse_date_string)
    df_dev = df_dev.dropna(subset=["Fecha_parsed"])
    df_dev["Year"]  = df_dev["Fecha_parsed"].dt.year
    df_dev["Month"] = df_dev["Fecha_parsed"].dt.month
    monthly = (
        df_dev.groupby(["Year", "Month"])["Cantidad"]
        .sum()
        .reset_index()
        .rename(columns={
            "Cantidad": f"{atb}_Devuelto"
        })
    )
    df = df.merge(
        monthly,
        left_on=["Año", "Mes"],
        right_on=["Year", "Month"],
        how="left"
    )
    df[f"{atb}_Devuelto"] = df[f"{atb}_Devuelto"].fillna(0)
    df.drop(columns=["Year", "Month"], inplace=True)

def smape(y_true, y_pred):

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / (denominator + 1e-6)) * 100

modelos = {
    "LinearRegression": LinearRegression(),
    "BayesianRidge":    BayesianRidge(),
    "RandomForest":     RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost":          XGBRegressor(random_state=42),
    "KNeighbors":       KNeighborsRegressor(),
    "SVR":              SVR(),
    "MLPRegressor":     MLPRegressor(hidden_layer_sizes=(50,), max_iter=10000, random_state=42)
}

resultados = []
for atb in atb_list:
    y = df[f"{atb}_Entregado"]
    for nombre, modelo in modelos.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X_base, y, test_size=0.2, random_state=42
        )
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        mae   = mean_absolute_error(y_test, y_pred)
        rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
        mape  = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-6))) * 100
        smape_val = smape(y_test, y_pred)
        r2    = r2_score(y_test, y_pred)

        resultados.append({
            "Antibiótico": atb,
            "Modelo":      nombre,
            "MAE":         round(mae, 2),
            "RMSE":        round(rmse, 2),
            "MAPE (%)":    round(mape, 2),
            "SMAPE (%)":   round(smape_val, 2),
            "R²":          round(r2, 2)
        })

df_res = pd.DataFrame(resultados)
print("\n📈 Métricas por modelo y antibiótico entregado a sala:")
print(df_res)

"""
plt.figure(figsize=(14, 6))
sns.barplot(data=df_res, x="Modelo", y="R²", hue="Antibiótico", palette="viridis")
plt.title("Comparación de R² por modelo y antibiótico")
plt.ylabel("R²")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""

plt.figure(figsize=(14, 6))
sns.barplot(data=df_res, x="Modelo", y="SMAPE (%)", hue="Antibiótico", palette="magma")
plt.title("Comparación de SMAPE por modelo y antibiótico")
plt.ylabel("SMAPE (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
