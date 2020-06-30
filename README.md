# thousandVoices
Text to Speech (TTS) with synthetical speaker embeddings

## Usage

## Model packaging (using MLflow)
Package the model using the artifacts in TTS/artifacts, and storing the package in packages/test_deployment

```
python TTS/pack.py TTS/artifacts packages/test_deployment
```
## Model deployment
Deploy the packaged model in packages/test_deployment using a Flask server

```
python TTS/deploy.py packages/test_deployment
```

## Testing
In your browser go to

[http://0.0.0.0:5000/TTS?S=Hello%20Stranger,%20how%20are%20you%20doing](http://0.0.0.0:5000/TTS?S=Hello%20Stranger,%20how%20are%20you%20doing)

Change REST endpoint is **/TTS**, taging an argument **S** containing the string to be synthesized.
The speaker embedding used for synthesis is selected randomly.