import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pyfunc
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
import shutil
import click
import tempfile
from pathlib import Path
# Load model module ... containing model creation and inference methods!
import model

# This will serve as an MLflow wrapper for the model
class ModelWrapper(mlflow.pyfunc.PythonModel):
    # Load in the model and all required artifacts
    # The context object is provided by the MLflow framework
    # It will contain all of the artifacts specified above
    def load_context(self, context):
        import torch
        import pickle

        # Load package meta
        with open(context.artifacts["meta"], 'rb') as handle:
            self.meta = pickle.load(handle)

        self.device = torch.device('cpu')

        # Load the model
        self.model = model.ThousandVoices(
            context.artifacts["model_state"],
            context.artifacts["model_dict"],
            context.artifacts["vocoder_state"],
            context.artifacts["vocoder_config"],
            context.artifacts["xvectors"],
            self.device)
 
    # Create a predict function for our models
    def predict(self, context, model_input):
        inference_type, args = model_input
        if inference_type == 'TTS':
            sentences, speaker = args
            with torch.no_grad():
                return self.model(sentences, speaker=speaker)


def pack_model(model_file: Path, addition_artifacts: dict, user_meta_data: Path, output_path: Path):
    # MLFlow is very picky about the output directory
    # Generate one temporal folder containing artifacts while generating the
    # mlflow package
    print(f'Writing packaged to {output_path} ...')
    shutil.rmtree(output_path, ignore_errors=True)
    #os.makedirs(f'{output_path}/tmp/', exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # Serialize the label encoder
        # This will be required at inference time
        #le_path = f'{tmp}/label_encoder.pkl'
        #with open(le_path, 'wb') as handle:
        #    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Build package meta dict
        package_meta = {
            'model_file': model_file,
            'user_meta': user_meta_data,
            }

        # Meta data about this package
        meta_path = f'{tmp}/meta.pkl'
        with open(meta_path, 'wb') as handle:
            pickle.dump(package_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Serialize the models state_dict
        #state_dict_path = f'{tmp}/state_dict.pt'
        #torch.save(model.state_dict(), state_dict_path)

        # Here we will create an artifacts object
        # It will contain all of the data artifacts that we want to package with the model
        artifacts = {
            "meta": meta_path,
        }
        artifacts.update(addition_artifacts)
        env = mlflow.pyfunc.get_default_conda_env() 

        # Package the model!
        mlflow.pyfunc.save_model(path=f'{output_path}',
                            python_model=ModelWrapper(),
                            artifacts=artifacts,
                            conda_env=env,
                            code_path=[model_file, user_meta_data])
        

def _pack(artifacts: Path, package: Path, user_meta: Path, model_file: Path):
    addition_artifacts = {
            "model_state": str(artifacts / "tacotron2/model.best"), 
            "model_config": str(artifacts / "tacotron2/model.json"), 
            "model_dict":  str(artifacts / "tacotron2/train_clean_460_units.txt"),
            "vocoder_state": str(artifacts / "parallelwavegan/model.pkl"),
            "vocoder_config":  str(artifacts / "parallelwavegan/config.yml"), 
            "xvectors": str(artifacts / "xvectors.pkl"),
    }

    print(f'Packing ...')
    pack_model(model_file, addition_artifacts, user_meta, package)


@click.command()
@click.argument('artifacts')
@click.argument('packagepath')
@click.option('--user_meta', default='TTS/meta.json', help='Path to the meta file')
@click.option('--model_file', default='TTS/model.py', help='Path to the python module containing the model')
def pack(artifacts, packagepath, user_meta, model_file):
    _pack(Path(artifacts), Path(packagepath), Path(user_meta), Path(model_file))


if __name__ == '__main__':
    pack()