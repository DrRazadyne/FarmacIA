import re
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import f_oneway, kruskal
import matplotlib.pyplot as plt
import seaborn as sns

cirugias_file = r"D:\CIRA\FarmacIA\Cirugias por año y por mes 2023 a 2025.xlsx"
salida_file   = r"D:\CIRA\FarmacIA\De sala a farmacia - ATB (_entradas_) del 01-01-23 al 30-06-25.xlsx"
entrada_file  = r"D:\CIRA\FarmacIA\De farmacia a sala - ATB (_salidas_) del 01-01-23 al 30-06-25.xlsx"

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

df_cir = pd.read_excel(cirugias_file, usecols="A:E")
df_cir.columns = ['Año', 'Mes', 'Planta', 'Guardia', 'Total']

ab_types = ['Ceftriaxona','Metronidazol','PTZ','Vancomicina','AMS']

def load_movimientos(path, tipo_label):
    dfs = []
    xls = pd.ExcelFile(path)
    for ab in ab_types:
        if ab not in xls.sheet_names:
            print(f"Omitiendo hoja '{ab}' en {path}")
            continue
        tmp = pd.read_excel(path, sheet_name=ab, header=None, usecols=[0,1])
        tmp.columns = ['Raw', 'Cantidad']
        tmp['Fecha']       = tmp['Raw'].apply(parse_date_string)
        tmp['Antibiotico'] = ab
        tmp['Tipo']        = tipo_label
        dfs.append(tmp[['Fecha','Antibiotico','Cantidad','Tipo']])
    return pd.concat(dfs, ignore_index=True)

df_sal = load_movimientos(salida_file,  tipo_label='salida')
df_ent = load_movimientos(entrada_file, tipo_label='entrada')

print(f"Salidas cargadas: {df_sal.shape}, Entradas cargadas: {df_ent.shape}")

df_all = pd.concat([df_sal, df_ent], ignore_index=True)
df_all = df_all.dropna(subset=['Fecha'])
df_all['Año'] = df_all['Fecha'].dt.year
df_all['Mes'] = df_all['Fecha'].dt.month

df_month = (
    df_all
      .pivot_table(
         index=['Año','Mes','Antibiotico'],
         columns='Tipo',
         values='Cantidad',
         aggfunc='sum',
         fill_value=0
      )
      .reset_index()
)

df_month.columns.name = None
df_month.columns = df_month.columns.str.strip().str.lower()

print("Columnas después del pivot:", df_month.columns.tolist())
assert 'salida' in df_month.columns,  "No se encontró la columna 'salida'"
assert 'entrada' in df_month.columns, "No se encontró la columna 'entrada'"

def month_to_season(m):
    if m in [12,1,2]:   return 'Verano'
    if m in [3,4,5]:    return 'Otoño'
    if m in [6,7,8]:    return 'Invierno'
    return 'Primavera'

df_month['estacion'] = df_month['mes'].apply(month_to_season)

results = []
for ab in ab_types:
    sub = df_month[df_month['antibiotico']==ab]
    groups = [
        sub.loc[sub['estacion']==est, 'salida'].values
        for est in ['Verano','Otoño','Invierno','Primavera']
    ]
    # omite tests si alguna estación no tiene datos
    if any(len(g)==0 for g in groups):
        print(f"  Atención: '{ab}' no tiene datos en todas las estaciones.")
        continue
    anova_p = f_oneway(*groups).pvalue
    kw_p    = kruskal(*groups).pvalue
    results.append({
        'antibiotico':     ab,
        'anova p-value':   anova_p,
        'kruskal p-value': kw_p
    })

df_results = pd.DataFrame(results)
print("\n===== Resultados estacionales =====")
print(df_results)

sns.set(style='whitegrid', palette='Set2')
for ab in ab_types:
    sub = df_month[df_month['antibiotico']==ab]
    if sub.empty: continue
    plt.figure(figsize=(6,4))
    sns.boxplot(
        data=sub,
        x='estacion', y='salida',
        order=['Verano','Otoño','Invierno','Primavera']
    )
    plt.title(f'Consumo mensual de {ab} por estación')
    plt.ylabel('Cantidad entregada')
    plt.tight_layout()
    plt.show()
