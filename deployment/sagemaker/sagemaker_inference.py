import json, os, joblib, numpy as np, pandas as pd
FEATURES=["month","lag1_cnt","lag12_cnt","avg_price"]
def model_fn(model_dir): return joblib.load(os.path.join(model_dir,'model.joblib'))
def input_fn(request_body, request_content_type):
    if request_content_type != 'application/json': raise ValueError(request_content_type)
    payload=json.loads(request_body); payload=[payload] if isinstance(payload,dict) else payload; df=pd.DataFrame(payload); return df[FEATURES]
def predict_fn(input_data, model): return np.clip(model.predict(input_data),0,20)
def output_fn(prediction, response_content_type): return json.dumps({'predictions': prediction.tolist()})
