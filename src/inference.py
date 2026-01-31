import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib

model = joblib.load("../artifacts/model.joblib")
X_test = pd.read_pickle("../data/inference/xtest.pkl")
test = pd.read_pickle("../data/inference/test.pkl")

pred_test = model.predict(X_test)
pred_test = np.clip(pred_test, 0, 20)

submission = pd.DataFrame({
    "ID": test["ID"],
    "item_cnt_month": pred_test
})

submission.to_csv("../data/predictions/submission.csv", index=False)