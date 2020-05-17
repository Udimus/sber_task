import os
import logging

from flask import Flask, request, jsonify, abort

from module.model import load_pipeline

PATH_TO_MODEL = os.getenv('MODEL', default='model.pkl.gz')

app = Flask(__name__)
model = load_pipeline(PATH_TO_MODEL)

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


@app.route('/ping')
def ping():
    return 'pong'


@app.route('/predict', methods=['POST'])
def predict():
    logger.info('Predict')

    data = request.get_json(force=True)
    is_list = isinstance(data, list)
    data = data if is_list else [data]
    logger.info(f'There is {len(data)} elements in request data.')

    predicts = model.predict(data).tolist()
    predicts = predicts if is_list else predicts[0]

    return jsonify(predicts)
