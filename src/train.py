import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib

# importamos los datos preparados
monthly = pd.read_pickle("../data/prep/monthly.pkl")
base = pd.read_pickle("../data/prep/base.pkl")

last_block = int(monthly["date_block_num"].max())

# Creación de variables simples, no dio tiempo de mucho se asumen que vendran mejoras
# adicional es algo similar a lo que planteamos: Nuestros planificadores usan métodos tradicionales—promedios móviles y ajustes manuales—que no pueden manejar la complejidad de 60 tiendas y 22,170 productos con patrones que varían por ubicación y estacionalidad. Ajustamos inventarios cada 14 días mientras la competencia lo hace en 48 horas.
lag1 = monthly[["date_block_num","shop_id","item_id","item_cnt_month"]].copy()
lag1["date_block_num"] += 1
lag1 = lag1.rename(columns={"item_cnt_month":"lag1_cnt"})
base = base.merge(lag1, on=["date_block_num","shop_id","item_id"], how="left")
base["lag1_cnt"] = base["lag1_cnt"].fillna(0)

# lag12 (estacionalidad anual)
lag12 = monthly[["date_block_num","shop_id","item_id","item_cnt_month"]].copy()
lag12["date_block_num"] += 12
lag12 = lag12.rename(columns={"item_cnt_month":"lag12_cnt"})
base = base.merge(lag12, on=["date_block_num","shop_id","item_id"], how="left")
base["lag12_cnt"] = base["lag12_cnt"].fillna(0)

# mes del año (0..11)
base["month"] = base["date_block_num"] % 12

# rellenamos con promedio histórico por item y luego mediana global
item_price = monthly.groupby("item_id")["avg_price"].mean()
base["avg_price"] = base["avg_price"].fillna(base["item_id"].map(item_price))
base["avg_price"] = base["avg_price"].fillna(monthly["avg_price"].median())


train_data = base[base["date_block_num"] <= last_block].copy()
train_data = train_data.dropna(subset=["item_cnt_month"]) 

X_cols = ["shop_id","item_id","item_category_id","month","lag1_cnt","lag12_cnt","avg_price"]

X = train_data[X_cols].astype(float)
y = train_data["item_cnt_month"].astype(float)

mask_tr = train_data["date_block_num"] < last_block
mask_va = train_data["date_block_num"] == last_block

X_tr, y_tr = X[mask_tr], y[mask_tr]
X_va, y_va = X[mask_va], y[mask_va]

model = Ridge(alpha=1.0, random_state=0)
model.fit(X_tr, y_tr)

pred_va = model.predict(X_va)
pred_va = np.clip(pred_va, 0, 20)

rmse = np.sqrt(mean_squared_error(y_va, pred_va))
print("RMSE valid (último mes):", rmse)

filename = "../artifacts/model.joblib"
joblib.dump(model, filename)
