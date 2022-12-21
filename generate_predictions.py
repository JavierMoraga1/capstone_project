import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import argparse

#Parameters
parser = argparse.ArgumentParser(description="Generate test predictions in a .csv file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-M","--model", help="model file")
parser.add_argument("-O","--output", help="output file")
args = parser.parse_args()
config = vars(args)

model_file = 'XGB_model.bin'
if config["model"] != None:
  model_file = config["model"]
output_file = 'predictions.csv'
if config["output"] != None:
  output_file = config["output"]
  
# Loading and preparing the dataset
print ('Loading and preparing the dataset...')

df_test = pd.read_csv('test.csv')

df_test.columns = df_test.columns.str.lower().str.replace(' ', '_')
df_test.cryosleep = pd.to_numeric(df_test.cryosleep, errors='coerce')
df_test.vip = pd.to_numeric(df_test.vip, errors='coerce')
categorical_columns = list(df_test.dtypes[df_test.dtypes == 'object'].index)
for c in categorical_columns:
    df_test[c] = df_test[c].str.lower().str.replace(' ', '_')

# Doing the predictions
print("Doing the predictions with model %s..." % model_file)
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

features = ['homeplanet', 'cryosleep', 'cabin', 'destination', 'age',
            'vip', 'roomservice', 'foodcourt', 'shoppingmall', 'spa', 'vrdeck']
X = dv.transform(df_test[features].to_dict(orient='records'))
features = dv.get_feature_names_out()
dX = xgb.DMatrix(X, feature_names=features)
y_pred = model.predict(dX)    

df_test['PassengerId'] = df_test['passengerid']
df_test['Transported'] = (y_pred >= 0.5)

df_test[['PassengerId','Transported']].to_csv(output_file,index=False)
print("File %s generated" % output_file)
