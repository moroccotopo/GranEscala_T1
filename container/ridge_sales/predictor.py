from __future__ import print_function

import json
from pathlib import Path

import flask
import joblib
import numpy as np
import pandas as pd

prefix = "/opt/ml/"
model_path = Path(prefix) / "model"


class ScoringService(object):
    model = None
    metadata = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = joblib.load(model_path / "model.joblib")
        return cls.model

    @classmethod
    def get_metadata(cls):
        if cls.metadata is None:
            cls.metadata = joblib.load(model_path / "metadata.joblib")
        return cls.metadata

    @classmethod
    def predict(cls, records):
        model = cls.get_model()
        metadata = cls.get_metadata()

        feature_columns = metadata["feature_columns"]
        clip_min = metadata["clip_min"]
        clip_max = metadata["clip_max"]

        data = pd.DataFrame(records)

        missing = [c for c in feature_columns if c not in data.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        X = data[feature_columns].astype(float)
        predictions = model.predict(X)
        predictions = np.clip(predictions, clip_min, clip_max)
        return predictions.tolist()


app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    try:
        health = ScoringService.get_model() is not None
        status = 200 if health else 404
    except Exception:
        status = 404

    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    if flask.request.content_type != "application/json":
        return flask.Response(
            response="This predictor only supports application/json",
            status=415,
            mimetype="text/plain",
        )

    payload = flask.request.get_json()

    if isinstance(payload, dict) and "instances" in payload:
        records = payload["instances"]
    elif isinstance(payload, list):
        records = payload
    else:
        return flask.Response(
            response="Formato inválido. Usa {'instances': [...]} o una lista.",
            status=400,
            mimetype="text/plain",
        )

    predictions = ScoringService.predict(records)
    result = json.dumps({"predictions": predictions})

    return flask.Response(response=result, status=200, mimetype="application/json")