FarmacIA: Modelos Predictivos de Consumo Antibiótico en una sala de Cirugía General

Este repositorio contiene los datos y scripts utilizados para el desarrollo, entrenamiento y evaluación de modelos predictivos de consumo antibiótico en una sala de Cirugia General. El objetivo es robustecer la gestión farmacéutica mediante inteligencia artificial aplicada a datos reales y simulados.

---

## 📁 Estructura del repositorio

### 📂 `/data/`
- `De farmacia a sala - ATB (_salidas_) del 01-01-23 al 30-06-25.xlsx`: Envio de antibioticos desde la farmacia a la sala, cada hoja tiene un antibiótico diferente.
- `De sala a farmacia - ATB (_entradas_) del 01-01-23 al 30-06-25.xlsx`: Devolucion de antibioticos desde la sala a la farmacia, cada hoja tiene un antibiótico diferente.
- `Cirugias por año y por mes 2023 a 2025.xlsx`: Registro mensual de cirugías por tipo.

### 📂 `/scripts/`
- `Analisis estadistico.py`: ANOVA, Kruskal-Wallis y visualización de patrones.
- `MultiOutputRegressor Test Data.py`: Entrenamiento con datos simulados, incluyendo devoluciones.
- `MultiOutputRegressor Real Data.py`: Pipeline completo con datos reales curados.

---

## ⚙️ Requisitos

- Python ≥ 3.8
- Bibliotecas: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

Instalación rápida:
```bash
pip install -r requirements.txt
