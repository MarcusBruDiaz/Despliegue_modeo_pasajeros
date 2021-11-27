import json

from pandas.core.indexes.base import Index
import numpy as np
import os
import pickle
import joblib
import pandas as pd

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It's the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION).
    # For multiple models, it points to the folder containing all deployed models (./azureml-models).
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'pasajero_model.pkl')
    model = joblib.load(model_path)
    
def run(raw_data):


    data = np.array(json.loads(raw_data)['data'])
    y_hat = model.predict(data)
 
    return json.dumps({"Prediccion_pasajeros": int(y_hat)})
