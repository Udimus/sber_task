import os
import logging

from flask import Flask, request, jsonify, abort
import pandas as pd

from module.model import load_pipeline
from module.prepare_data import (
    BIN_FEATURES,
    CAT_FEATURES,
    NUM_FEATURES,
)

PATH_TO_MODEL = os.getenv('MODEL', default='model.pkl.gz')
FEATURES = BIN_FEATURES + CAT_FEATURES + NUM_FEATURES

app = Flask(__name__)
model = load_pipeline(PATH_TO_MODEL)

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


def abort_request(message, code=400):
    logger.error('Request error (%d): %s', code, message)
    response = jsonify(code=code, message=message)
    response.status_code = code
    abort(response)


def prepare_data(data):
    is_list = isinstance(data, list)
    data = data if is_list else [data]
    for data_element in data:
        if isinstance(data_element, dict):
            for feature in FEATURES:
                if feature not in data_element:
                    abort_request(f"Data should have field '{feature}'.")
        else:
            abort_request('Data should be dict or list of dicts.')
    data = pd.DataFrame(data)
    logger.info(f'There is {len(data)} elements in request data.')
    return is_list, data


@app.route('/ping')
def ping():
    return 'pong'


@app.route('/predict', methods=['POST'])
def predict():
    logger.info('Predict')
    data = request.get_json(force=True)
    is_list, data = prepare_data(data)

    try:
        predicts = model.predict(data).tolist()
        predicts = predicts if is_list else predicts[0]
        return jsonify(predicts)
    except Exception as e:
        abort_request(e, code=500)
