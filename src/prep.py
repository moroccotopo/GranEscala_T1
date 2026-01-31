import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

DATA = Path("../data/raw")
files = [
    "sales_train.csv",
    "test.csv",
    "items.csv",
    "item_categories.csv",
    "shops.csv",
    "sample_submission.csv",
]

dfs = {}
for f in files:
    p = DATA / f
    if p.exists():
        dfs[f] = pd.read_csv(p)
    else:
        print("No encontré:", p)

list(dfs.keys())

train = dfs["sales_train.csv"].copy()
test  = dfs["test.csv"].copy()
items = dfs["items.csv"][["item_id","item_category_id"]].copy()

# agración mensual
monthly = (train
           .groupby(["date_block_num","shop_id","item_id"], as_index=False)
           .agg(item_cnt_month=("item_cnt_day","sum"),
                avg_price=("item_price","mean")))

monthly["item_cnt_month"] = monthly["item_cnt_month"].clip(0, 20)

monthly = monthly.merge(items, on="item_id", how="left")

last_block = int(monthly["date_block_num"].max())
test_block= last_block+1
test_base = test.copy()
test_base["date_block_num"] = test_block

base = pd.concat([
    monthly[["date_block_num","shop_id","item_id","item_cnt_month","avg_price","item_category_id"]],
    test_base.merge(items, on="item_id", how="left")[["date_block_num","shop_id","item_id","item_category_id"]]
], ignore_index=True, sort=False)


base.to_pickle("../data/prep/base.pkl")