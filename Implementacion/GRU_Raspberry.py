
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
#from google.colab import files #import csv file(Dataset)
#files.upload()

print("Vamos a importar el modelo")
from keras.models import load_model
model = load_model("model.h5")
model.summary()
print("Todo salio de maravilla")

# READ CSV
pd.options.display.max_columns=150
# Read in the paciente data csv - keep only the columns we need, in this case: Time, glucose, intake e insulin
df=pd.read_csv('A9.csv', encoding='utf-8', usecols=['Carbo','Tiempo_F','Tiempo','Glucosa','Ingesta','Insulina'])
# Show a snaphsot of data
dfsn=df.copy()

df['Glucosa'] = (df['Glucosa'] - df['Glucosa'].min()) / (df['Glucosa'].max() - df['Glucosa'].min())
df['Insulina'] = (df['Insulina'] - df['Insulina'].min()) / (df['Insulina'].max() - df['Insulina'].min())

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
print("Base de datos leida correctamente")

# Define Number of fetures
n_features = 2
n_steps_in=15

Num_ingesta="42" # Acá defina el número de la ingesta a predecir!
Ingesta="Ingesta"+ Num_ingesta


xi1=array(np.array(df3[Ingesta][0:n_steps_in]))
xi2=array(np.array(df4[Ingesta][0:n_steps_in]))

import time
import RPi.GPIO as GPIO
PULSADOR_PIN = 18
LED_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(PULSADOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED_PIN, GPIO.OUT)
#Crear matriz vacia
matriz=np.empty((15,2))

print("Le agregamos el pulsador")
for i in range(n_steps_in):
    input("Presine el pulsador para tomar medidas")
    #while GPIO.input(PULSADOR_PIN) == GPIO.HIGH:
     #   pass
    matriz[i]=[xi1[i],xi2[i]]
    print(matriz,"->Insulina y carbohidratos en el minuto",i+1)


    # Esperar hasta que se libere el pulsador
    #while GPIO.input(PULSADOR_PIN) == GPIO.LOW:
     #   pass
    # Encender el LED durante 1 segundo cuando se suelta el pulsador
    #GPIO.output(LED_PIN, GPIO.HIGH)
    time.sleep(1)
    #GPIO.output(LED_PIN, GPIO.LOW)


print("\n\nIngesta luego de 15 minutos\n",matriz)

# Limpiar los pines GPIO antes de salir
GPIO.cleanup()
#XIN = np.append(xi1, xi2,axis=0)
#matriz = np.array(list(zip(xi1,xi2)))
x_input1=matriz.reshape(n_steps_in,2)
x_input = x_input1.reshape((1, n_steps_in, n_features))

print(x_input)

yhat = model.predict(x_input, verbose=0)
yhat = (yhat* (dfsn['Glucosa'].max() - dfsn['Glucosa'].min())) + dfsn['Glucosa'].min()


N_grap=n_steps_in
plt.figure(figsize=(6, 4))
plt.plot(array(np.array(df3sn[Ingesta][0:N_grap+430])), label='Glucosa de Entrada')
plt.plot(range(N_grap,N_grap+430),array(np.array(df3sn[Ingesta][N_grap:N_grap+430])), label=f'Glucosa real {Num_ingesta}')
plt.plot(range(n_steps_in,n_steps_in+430),array(np.array(yhat.reshape(-1,1))), label=f'Predicción {Num_ingesta}')
plt.xlabel("Tiempo (min)") # título del eje x
plt.ylabel("Nivel de glucosa (mg/dL)") # título del eje y
plt.legend()
#plt.suptitle(( f"Parámetros: Dropopt={D}, Units={U}, Batch_size={B}, Epochs={E}, Folders={F}"), fontsize=10)
plt.tight_layout() # ajusta el espaciado de la gráfica
#plt.text(0, dfsn['Glucosa'].max()*0.25, f"Error= {round(Err, 4)}%", ha='center', va='center')


plt.show()

i=0
for glu in yhat:
    i=i+1
    if glu>=180:
        print("¡Alerta! A partir del minuto ",i, "va a ocurrir un evento de hiperglucemia")
        break
    elif glu<70:
        print("¡Alerta! En el minuto ",i, "Va a ocurrir un evento de hipoglucemia")


