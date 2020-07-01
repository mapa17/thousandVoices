from flask import Flask, request, jsonify, send_file, after_this_request, Response
import mlflow.pyfunc
import pandas as pd
import json
import click
import pickle
from matplotlib import pyplot as plt
import os
import torch
import torchvision.utils as vutils

import tempfile
import tarfile
import shutil
from PIL import Image

from IPython.display import display, Audio
import base64
#import shutil

# Name of the apps module package
app = Flask(__name__)
flask_env = {
    'model': None,
    'user_meta': None
}


# Meta data endpoint
@app.route('/', methods=['GET'])
def meta_data():
	return jsonify(flask_env['user_meta'])


## Test in browser by going to
# http://localhost:5000/TTS?S=Hello%20Bob,%20how%20are%20you%20doing

# Can be called with /TTS?S=SomeText
@app.route("/TTS")
def TTS():
    sentence = [str(request.args.get('S')), ]
    print(f'Recieved Text: {sentence}')

    speaker = None

    speach = flask_env['model'].predict(('TTS', (sentence, speaker)))
    a = Audio(speach[0].view(-1).cpu().numpy(), rate=24000)
    result = Response(a.data, mimetype="audio/x-wav")

    return result

def _deploy(package: str):
    print(f'Deploying package {package} ...')
    # Load in the model at app startup
    model = mlflow.pyfunc.load_model(package)

    # Load package metadata
    with open(model._model_impl.context.artifacts["meta"], 'rb') as handle:
        meta = pickle.load(handle)

    # Load user metadata
    with open(f"{package}/code/{os.path.basename(meta['user_meta'])}", 'rb') as handle:
        user_meta = json.load(handle)

    flask_env['model'] = model
    flask_env['meta'] = meta
    flask_env['user_meta'] = user_meta

    return app


@click.command()
@click.argument('package', type=click.Path(exists=True))
@click.option('--port', default=5000, help='Port to serve model')
@click.option('--debug', default=False, help='Enable flask debuggin', is_flag=True)
def deploy(package: str, port: int, debug: bool):
    app = _deploy(package)
    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    deploy()