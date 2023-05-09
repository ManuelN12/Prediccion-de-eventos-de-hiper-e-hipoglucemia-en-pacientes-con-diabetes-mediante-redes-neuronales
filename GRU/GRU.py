# -*- coding: utf-8 -*-
"""Copy of   P1 Multivariable RYA2DT1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vDPg_FTqLak3jcaCS2nduVxNpREIP4ph
"""

import os 
from google.colab import drive


# univariate multi-step vector-output stacked lstm example

import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU, Dropout, BatchNormalization
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from numpy import hstack
from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets, metrics

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Tensorflow / Keras
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Bidirectional, GRU, RepeatVector, Dense, TimeDistributed # for creating layers inside the Neural Network

# Data manipulation
import pandas as pd # for data manipulation
print('pandas: %s' % pd.__version__) # print version
import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version

# Sklearn
import sklearn
print('sklearn: %s' % sklearn.__version__) # print version
from sklearn.preprocessing import MinMaxScaler # for feature scaling

# Visualization
import matplotlib.pyplot as plt
import plotly 
import plotly.express as px
import plotly.graph_objects as go
print('plotly: %s' % plotly.__version__) # print version
from google.colab import files #import csv file(Dataset) 
files.upload()

# READ CSV 
pd.options.display.max_columns=150

# Read in the paciente data csv - keep only the columns we need, in this case: Time, glucose, intake e insulin
df=pd.read_csv('A9.csv', encoding='utf-8', usecols=['Carbo','Tiempo_F','Tiempo','Glucosa','Ingesta','Insulina'])

# Show a snaphsot of data
dfsn=df.copy()

df['Glucosa'] = (df['Glucosa'] - df['Glucosa'].min()) / (df['Glucosa'].max() - df['Glucosa'].min())
df['Insulina'] = (df['Insulina'] - df['Insulina'].min()) / (df['Insulina'].max() - df['Insulina'].min())
print(min(df['Glucosa']))
print(min(df['Glucosa']))
print(max(df['Glucosa']))
print(max(df['Glucosa']))

# Create a copy of an original dataframe
df2=df[['Carbo','Tiempo','Ingesta','Insulina','Glucosa' ]].copy()
df2t=df[['Carbo','Tiempo_F','Ingesta','Insulina','Glucosa' ]].copy()
df2sn=dfsn[['Carbo','Tiempo','Ingesta','Insulina','Glucosa' ]].copy()

# Transpose dataframe 

df2_pivot=df2.pivot(index=['Carbo'], columns='Tiempo')['Glucosa']
df2_pivot1=df2.pivot(index=['Carbo'], columns='Tiempo')['Insulina']
df2_pivot2=df2.pivot(index=['Carbo'], columns='Tiempo')['Ingesta']

df2t_pivot=df2t.pivot(index=['Carbo'], columns='Tiempo_F')['Glucosa']
############################### Transpose dataframe 

df2sn_pivot=df2sn.pivot(index=['Carbo'], columns='Tiempo')['Glucosa']
df2sn_pivot1=df2sn.pivot(index=['Carbo'], columns='Tiempo')['Insulina']
df2sn_pivot2=df2sn.pivot(index=['Carbo'], columns='Tiempo')['Ingesta']

df3=np.transpose(df2_pivot)
df4=np.transpose(df2_pivot1)
##############################
df3sn=np.transpose(df2sn_pivot)
df4sn=np.transpose(df2sn_pivot1)

# Plot levels of Glucose for each patient!
fig = go.Figure()
for Paciente in df2_pivot.index:
    fig.add_trace(go.Scatter(x=df2_pivot.loc[Paciente, :].index, 
                             y=df2_pivot.loc[Paciente, :].values,
                             mode='lines',
                             name=Paciente,
                             opacity=0.8,
                             line=dict(width=1)
                            ))

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'), showlegend=True)

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='Date'
                )

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='mg'
                )

# Set figure title
fig.update_layout(title=dict(text="Glucose behavior", font=dict(color='black')))

fig.show()

# Plot levels of Glucose for each patient!
fig = go.Figure()
for Paciente in df2t_pivot.index:
    fig.add_trace(go.Scatter(x=df2t_pivot.loc[Paciente, :].index, 
                             y=df2t_pivot.loc[Paciente, :].values,
                             mode='lines',
                             name=Paciente,
                             opacity=0.8,
                             line=dict(width=1)
                            ))

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'), showlegend=True)

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='Date'
                )

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='mg'
                )

# Set figure title
fig.update_layout(title=dict(text="Glucose behavior", font=dict(color='black')))

fig.show()

##### split a univariate sequence into samples #################################
def split_sequence(sequence,seq2, n_steps_in, n_steps_out):
	X, I, y = list(),list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_i, seq_y = sequence[i:end_ix],seq2[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		I.append(seq_i)
		y.append(seq_y)    
	return array(X), array(I),array(y)
################################################################################
N=38                                                                                         #10% pruebas,60%train 30% Validación
raw_seq = array(np.array(df['Glucosa'][0:480*N])) #de 0 a 480 are all dataa for i intake 
raw_seq2 = array(np.array(df['Insulina'][0:480*N])) 
# choose a number of input and output
n_steps_in, n_steps_out = 15, 430                                                 
# split into samples
X, I, y = split_sequence(raw_seq, raw_seq2, n_steps_in, n_steps_out)
# Concatenar las matrices de glucosa e insulina a lo largo del eje de características
X = np.concatenate((X[:, :, np.newaxis], I[:, :, np.newaxis]), axis=2)
print('shape X =',X.shape)
print('shape X =',y.shape)

#split data test and train.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
                                         
print('shape X_train =',X_train.shape)
print('shape y_tain =',y_train.shape)
print('shape X_test =',X_test.shape)
print('shape y_test =',y_test.shape)

# Commented out IPython magic to ensure Python compatibility.
# Define Number of fetures 
n_features = X.shape[2]   
D=0
U=64
F=5
E=5
B=32     
import datetime
# %load_ext tensorboard
# Crear objeto TensorBoard para guardar los registros
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = keras.callbacks.EarlyStopping(monitor='loss', patience=4)

# define model  tanh sigmoid relu
model = keras.Sequential()
model.add(GRU(n_steps_in, activation='relu',  recurrent_activation='sigmoid',return_sequences=True, input_shape=(X.shape[1], X.shape[2]))) 
#model.add(Dropout(D))
model.add(GRU(U, activation="relu", recurrent_activation='sigmoid', return_sequences=False))
#model.add(Dropout(D))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mae',metrics=['MeanSquaredError','RootMeanSquaredError'])
model.summary()
callbacks = keras.callbacks.EarlyStopping(monitor='loss', patience=4)

# Ejecutar la validación cruzada
kfold = KFold(n_splits=F, shuffle=True)
# Crear una lista para almacenar los objetos de historia
history_list = []
for train, test in kfold.split(X_train, y_train):
  # Ajustar el modelo y almacenar el historial en la lista
  history = model.fit(X_train, y_train, batch_size=B, epochs=E, callbacks=callbacks, verbose=1)
  history_list.append(history)
  # Evaluar el modelo
  scores = model.evaluate(X_test, y_test, verbose=1)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print(scores)

# Graficar las curvas de pérdida y métricas en todas las iteraciones
plt.figure(figsize=(20,4))
for i, history in enumerate(history_list):
  plt.subplot(1, F, i+1)
  plt.plot(history.history['loss'], label='train_loss')
 # plt.plot(history.history['val_loss'], label='val_loss')
  plt.title(f'Fold {i+1}')
  plt.legend()
plt.show()

# Graficar las curvas de pérdida y métricas en todas las iteraciones
plt.figure(figsize=(5,4))
for i, history in enumerate(history_list):
  plt.plot(1, F, i+1)
  plt.plot(history.history['loss'], label=('train_loss '+ str(i)))
 # plt.plot(history.history['val_loss'], label='val_loss')
  plt.title('Error por Fold')
  plt.xlim(0, E)
  plt.ylim(0.025, 0.125)
  # Agregar etiquetas a los ejes
  plt.xlabel('Épocas')
  plt.ylabel('Pérdida')
  plt.legend()

plt.show()

model.save("model9.h5")

# Define Number of fetures 
n_features = X.shape[2]                                                     

Num_ingesta="39" # Acá defina el número de la ingesta a predecir!
Ingesta="Ingesta"+ Num_ingesta 
xi1=array(np.array(df3[Ingesta][0:n_steps_in]))
xi2=array(np.array(df4[Ingesta][0:n_steps_in]))

#XIN = np.append(xi1, xi2,axis=0)
matriz = np.array(list(zip(xi1,xi2)))
x_input1=matriz.reshape(n_steps_in,2)
x_input = x_input1.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
yhat = (yhat* (dfsn['Glucosa'].max() - dfsn['Glucosa'].min())) + dfsn['Glucosa'].min()

N_grap=n_steps_in
area_p = np.sum(yhat)
print("El área bajo la prediccion es",area_p)
real=np.array(df3sn[Ingesta][N_grap:N_grap+n_steps_out])
area_r=np.sum(real)
print("El área bajo la curva real es: ",area_r)
Err=((area_p-area_r)/area_r)*100
print("El error calculado es: ", Err, "%")

plt.figure(figsize=(6, 4))
plt.plot(array(np.array(df3sn[Ingesta][0:N_grap+n_steps_out])), label='Glucosa de Entrada')                       
plt.plot(range(N_grap,N_grap+n_steps_out),array(np.array(df3sn[Ingesta][N_grap:N_grap+n_steps_out])), label=f'Glucosa real {Num_ingesta}')  
plt.plot(range(N_grap,N_grap+n_steps_out),array(np.array(yhat.reshape(-1,1))), label=f'Predicción {Num_ingesta}')
plt.xlabel("Tiempo (min)") # título del eje x
plt.ylabel("Nivel de glucosa (mg/dL)") # título del eje y
plt.legend()
plt.suptitle(( f"Parámetros: Dropopt={D}, Units={U}, Batch_size={B}, Epochs={E}, Folders={F}"), fontsize=10)
plt.tight_layout() # ajusta el espaciado de la gráfica
plt.text(0, dfsn['Glucosa'].max()*0.15, f"Error= {round(Err, 4)}%", ha='center', va='center')
plt.show()