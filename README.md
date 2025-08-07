# 🧠 FarmacIA: Modelos Predictivos de Consumo Antibiótico en una Sala de Cirugía General

Este repositorio contiene los datos y scripts utilizados para el desarrollo, entrenamiento y evaluación de modelos predictivos de consumo antibiótico en una sala de Cirugía General. El objetivo es robustecer la gestión farmacéutica mediante inteligencia artificial aplicada a datos reales y simulados, integrando variables clínicas, estacionales y logísticas.

---

## 📌 Objetivos del estudio

- Predecir el consumo mensual por antibiótico en función de variables quirúrgicas y temporales.
- Comparar el rendimiento de modelos tradicionales y redes neuronales densas.
- Simular variables faltantes (como devoluciones) para robustecer el pipeline.
- Documentar cada decisión metodológica para facilitar la transferencia hospitalaria.
- Proponer un marco reproducible para la gestión inteligente de recursos farmacéuticos.

---

## 📁 Estructura del repositorio

### 📂 `/data/`
- `De farmacia a sala - ATB (_salidas_) del 01-01-23 al 30-06-25.xlsx`: Envío de antibióticos desde farmacia a sala. Cada hoja corresponde a un antibiótico.
- `De sala a farmacia - ATB (_entradas_) del 01-01-23 al 30-06-25.xlsx`: Devolución de antibióticos desde sala a farmacia. Cada hoja corresponde a un antibiótico.
- `Cirugías por año y por mes 2023 a 2025.xlsx`: Registro mensual de cirugías por tipo.

### 📂 `/scripts/`
- `Analisis estadistico.py`: ANOVA, Kruskal-Wallis y visualización de patrones de consumo.
- `MultiOutputRegressor Test Data.py`: Entrenamiento con datos simulados, incluyendo devoluciones.
- `MultiOutputRegressor Real Data.py`: Pipeline completo con datos reales curados.

---

## ⚙️ Requisitos

- Python ≥ 3.8
- Bibliotecas principales:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Instalación rápida:
```bash
pip install -r requirements.txt
```

---

## 🚀 Ejecución
Para ejecutarlo yo uso pycharm, cambiar la carpeta donde se encuentran los archivos de excel y ejecutar el script