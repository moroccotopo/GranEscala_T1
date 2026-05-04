from __future__ import annotations
import argparse, json, os, shutil, tarfile
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
FEATURES = ["month", "lag1_cnt", "lag12_cnt", "avg_price"]

def log(x): print(f"[{datetime.now(timezone.utc).isoformat()}] {x}", flush=True)
def mkdir(p): Path(p).mkdir(parents=True, exist_ok=True); return Path(p)

def build_monthly(raw_dir: Path) -> pd.DataFrame:
    sales = pd.read_csv(raw_dir / "sales_train.csv")
    monthly = sales.groupby(["date_block_num","shop_id","item_id"], as_index=False).agg(item_cnt_month=("item_cnt_day","sum"), avg_price=("item_price","mean"))
    monthly["item_cnt_month"] = monthly["item_cnt_month"].clip(0,20)
    items = raw_dir / "items.csv"
    if items.exists():
        it = pd.read_csv(items)[["item_id","item_category_id"]].drop_duplicates("item_id")
        monthly = monthly.merge(it, on="item_id", how="left")
    else:
        monthly["item_category_id"] = -1
    monthly = monthly.sort_values(["shop_id","item_id","date_block_num"])
    monthly["lag1_cnt"] = monthly.groupby(["shop_id","item_id"])["item_cnt_month"].shift(1).fillna(0)
    monthly["lag12_cnt"] = monthly.groupby(["shop_id","item_id"])["item_cnt_month"].shift(12).fillna(0)
    monthly["avg_price"] = monthly["avg_price"].fillna(monthly["avg_price"].median())
    monthly["month"] = monthly["date_block_num"] % 12
    monthly["item_category_id"] = monthly["item_category_id"].fillna(-1).astype(int)
    return monthly

def preprocess(input_raw, output_prep):
    out = mkdir(output_prep); df = build_monthly(Path(input_raw)); df.to_parquet(out / "monthly.parquet", index=False); log(f"preprocess rows={len(df)}")

def train(input_prep, output_model):
    out = mkdir(output_model); df = pd.read_parquet(Path(input_prep)/"monthly.parquet")
    val_month = int(df.date_block_num.max()); tr = df[df.date_block_num < val_month]; va = df[df.date_block_num == val_month]
    if tr.empty: tr, va = df, df
    model = Ridge(alpha=1.0).fit(tr[FEATURES], tr["item_cnt_month"])
    pred = np.clip(model.predict(va[FEATURES]),0,20)
    meta = {"features": FEATURES, "validation_month": val_month, "rmse": float(np.sqrt(np.mean((va.item_cnt_month.to_numpy() - pred) ** 2))), "mae": float(mean_absolute_error(va.item_cnt_month, pred)), "created_at": datetime.now(timezone.utc).isoformat()}
    joblib.dump(model, out/"model.joblib"); (out/"metadata.json").write_text(json.dumps(meta, indent=2))
    model_dir = mkdir(out/"model"); shutil.copy2(out/"model.joblib", model_dir/"model.joblib"); (model_dir/"metadata.json").write_text(json.dumps(meta, indent=2))
    inf = Path('/opt/program/deployment/sagemaker/sagemaker_inference.py')
    if inf.exists():
        code = mkdir(model_dir/"code"); shutil.copy2(inf, code/"inference.py")
    with tarfile.open(out/"model.tar.gz", "w:gz") as tar:
        tar.add(model_dir/"model.joblib", arcname="model.joblib"); tar.add(model_dir/"metadata.json", arcname="metadata.json")
        if (model_dir/"code/inference.py").exists(): tar.add(model_dir/"code/inference.py", arcname="code/inference.py")
    log(meta)

def metric_dict(y, p):
    denom = float(np.abs(y).sum()); return {"mae": float(mean_absolute_error(y,p)), "rmse": float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2))), "wape": float(np.abs(y-p).sum()/denom) if denom else None, "bias": float((p-y).sum()/denom) if denom else None, "n_obs": int(len(y))}

def evaluate(input_prep, input_model, output_eval, run_id):
    out = mkdir(output_eval); df = pd.read_parquet(Path(input_prep)/"monthly.parquet"); model = joblib.load(Path(input_model)/"model.joblib")
    val_month = int(df.date_block_num.max()); ev = df[df.date_block_num == val_month].copy(); ev["run_id"] = run_id; ev["y_true"] = ev.item_cnt_month.astype(float); ev["y_pred"] = np.clip(model.predict(ev[FEATURES]),0,20); ev["error"] = ev.y_pred-ev.y_true; ev["abs_error"] = abs(ev.error)
    eval_dir = mkdir(out/"forecast_evaluation")
    ev[["run_id","date_block_num","shop_id","item_id","item_category_id","y_true","y_pred","error","abs_error"]].to_parquet(eval_dir/"part-000.parquet", index=False)
    rows = [{"run_id":run_id,"level":"global","group_id":"all", **metric_dict(ev.y_true, ev.y_pred)}]
    for level, col in [("category","item_category_id"),("product","item_id")]:
        for k,g in ev.groupby(col): rows.append({"run_id":run_id,"level":level,"group_id":str(k), col:int(k), **metric_dict(g.y_true, g.y_pred)})
    met = pd.DataFrame(rows); metrics_dir = mkdir(out/"forecast_metrics"); met.to_parquet(metrics_dir/"part-000.parquet", index=False)
    report = {"run_id": run_id, "metrics": rows[0], "validation_month": val_month}; (out/"evaluation_report.json").write_text(json.dumps(report, indent=2)); log(report)

def batch_predict(input_prep, input_model, output_curated, run_id):
    out = mkdir(output_curated); df = pd.read_parquet(Path(input_prep)/"monthly.parquet"); model = joblib.load(Path(input_model)/"model.joblib")
    last = int(df.date_block_num.max()); base = df[df.date_block_num == last].copy(); base["run_id"] = run_id; base["forecast_month"] = str(last+1); base["prediction"] = np.clip(model.predict(base[FEATURES]),0,20)
    pred_dir = mkdir(out/"forecast_predictions")
    base[["run_id","forecast_month","shop_id","item_id","item_category_id","prediction"]].to_parquet(pred_dir/"part-000.parquet", index=False); log(f"pred rows={len(base)}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--step", required=True); ap.add_argument("--input-raw", default="/opt/ml/processing/input/raw"); ap.add_argument("--input-prep", default="/opt/ml/processing/input/prep"); ap.add_argument("--input-model", default="/opt/ml/processing/input/model"); ap.add_argument("--output-prep", default="/opt/ml/processing/output/prep"); ap.add_argument("--output-model", default="/opt/ml/processing/output/model"); ap.add_argument("--output-eval", default="/opt/ml/processing/output/evaluation"); ap.add_argument("--output-curated", default="/opt/ml/processing/output/curated"); ap.add_argument("--run-id", default=os.environ.get("RUN_ID", datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"))); args=ap.parse_args()
    if args.step=="preprocess": preprocess(args.input_raw,args.output_prep)
    elif args.step=="train": train(args.input_prep,args.output_model)
    elif args.step=="evaluate": evaluate(args.input_prep,args.input_model,args.output_eval,args.run_id)
    elif args.step=="batch_predict": batch_predict(args.input_prep,args.input_model,args.output_curated,args.run_id)
    else: raise ValueError(args.step)
if __name__ == "__main__": main()
