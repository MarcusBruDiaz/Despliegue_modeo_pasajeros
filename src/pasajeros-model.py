import pandas as pd
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#Modelos
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.compose import ColumnTransformer
# SPLIT DE LOS DATOS 
from sklearn.model_selection import train_test_split

import multiprocessing

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import pickle


# Comvertir de series de tiempo a aprendizaje supervisado
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg



# Se lee el dataset  y se elimninan columnas irrelebantes para el modelo
df= pd.read_csv('Dataset_pasajeros_TT_norte_Medellin_limpio.csv', delimiter=",")
df.columns=['FECHA_DESPACHO','TERMINAL','CLASE_VEHICULO','NIVEL_SERVICIO','MUNICIPIO_ORIGEN_RUTA','MUNICIPIO_DESTINO_RUTA','HORA_DESPACHO','TIPO_DESPACHO','DESPACHOS','PASAJEROS']
df=df.drop(["TERMINAL","CLASE_VEHICULO","NIVEL_SERVICIO","TIPO_DESPACHO","MUNICIPIO_ORIGEN_RUTA","MUNICIPIO_DESTINO_RUTA","HORA_DESPACHO"], axis=1)



#Preprocesamiento 
#Se cambia formato columna fecha
df["FECHA_DESPACHO"]= pd.to_datetime(df["FECHA_DESPACHO"])

#Se establece como indice la columna fecha
df = df.set_index('FECHA_DESPACHO')


# Se filtra los registros a parir de octubre de 2020
df= df["2020-10":]

# Se agrugan los registros por dias 
df=df.resample('D').sum()
#print(df.head())
# Se cambia el formato a enteror
df['PASAJEROS']=pd.to_numeric(df['PASAJEROS'], downcast='integer')
df['DESPACHOS']=pd.to_numeric(df['DESPACHOS'], downcast='integer')


# Se combierten los 7 dias de resagos en columnas 
step_back = 7
df = series_to_supervised(df,step_back,1)
#print(df.head())
df=df.rename(columns={'var2(t)':'PASAJEROS'})

X= df.drop(['PASAJEROS'],axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
X[:] = scaler.fit_transform(X)
X["PASAJEROS"] = df["PASAJEROS"]



#DIVISION DE LOS DATOS
Train, Test = train_test_split(df, test_size = 0.20, shuffle = False)


# SE CREAN LOS DATOS DE ENTRENAMIENTO Y TEST 
X_Train = Train.drop(columns='PASAJEROS')
y_Train = Train['PASAJEROS']
X_Test= Test.drop(columns='PASAJEROS')
y_Test= Test['PASAJEROS']


train_scores = []
models = [] 
mae = [] 
 
for KF in range(2,20):
    lin3 = LinearRegression()
    linea= lin3.fit(X_Train,y_Train)
    kf = KFold(n_splits=KF)
    scores = cross_val_score(linea, X_Train, y_Train, cv=kf) #pasamos el modelo entrenado, x train, y train y kfold para cada iteraciion
    train_scores.append(scores.mean())
    models.append(lin3)
    y_pred2=lin3.predict(X_Test)
    mae.append(mean_absolute_error(y_Test,y_pred2)) 
 

#ML Model: Model Selection
#lr=LinearRegression()
#lr.fit(X_Train,y_Train)

#predictions 
#pred2=lr.predict(X_Test)

print(mean_absolute_error(y_Test,y_pred2))

#Registro
with open('./outputs/pasajero_model.pkl', 'wb') as model_pkl:
    pickle.dump(lin3, model_pkl)
