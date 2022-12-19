import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

#from flask import Flask
#from flask import request
#from flask import jsonify

model_file = 'XGB_model.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

features = ['homeplanet', 'cryosleep', 'cabin', 'destination', 'age',
            'vip', 'roomservice', 'foodcourt', 'shoppingmall', 'spa', 'vrdeck']

class Passenger(BaseModel):
    homeplanet: Optional[str] = None
    cryosleep: Optional[float] = None
    cabin: Optional[str]= None
    destination: Optional[str] = None
    age: Optional[float]= None
    vip: Optional[float] = None
    roomservice: Optional[float] = None
    foodcourt: Optional[float] = None
    shoppingmall: Optional[float] = None
    spa: Optional[float] = None
    vrdeck: Optional[float] = None

    """
    homeplanet: str | None = None
    cryosleep: float | None = None
    cabin: str | None = None
    destination: str | None = None
    age: float | None = None
    vip: float | None = None
    roomservice: float | None = None
    foodcourt: float | None = None
    shoppingmall: float | None = None
    spa: float | None = None
    vrdeck: float | None = None
    """

    class Config:
      schema_extra = {
        "example": {
          "homeplanet": "europa",
          "cryosleep": 0,
          "cabin": "e/608/s",
          "destination": "55_cancri_e",
          "age": 32,
          "vip": 0,
          "roomservice": 0,
          "foodcourt": 1049,
          "shoppingmall": 0,
          "spa": 353,
          "vrdeck": 3235
        }            
      }

#app = Flask('trip_duration')

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello World"}

#@app.route('/predict', methods=['POST'])
@app.post("/predict/")
async def predict(passenger: Passenger):

    passenger_dict = passenger.dict()
    X = dv.transform(passenger_dict)
    dX = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
    y_pred = model.predict(dX)

    result = {
        'Survival': int(y_pred)
    }
    
    return result
"""
    trip = request.get_json(force=True)
    print(trip)

    df_request = pd.DataFrame.from_dict([trip])
  
    features = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
              'dropoff_longitude', 'dropoff_latitude', 'month',
              'weekday', 'hour', 'hdistance', 'cdirection']

    X = df_request[features].values
    dX = xgb.DMatrix(X, feature_names=features)
    y_pred = np.exp(model.predict(dX)) + 1

    print(y_pred)
    result = {
        'trip_duration': int(y_pred)
    }
"""
    #return jsonify(result)

#if __name__ == "__main__":
#    app.run(debug=True, host='0.0.0.0', port=port)