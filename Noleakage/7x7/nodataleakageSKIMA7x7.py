import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python.client import device_lib
import time as tt
import os

#NOMENCLATURA
# _u --> unscaled (scaler.inverse_transform)
# _s --> static
# _o --> online learning
# nada --> scaled (scaler.transform)




# GET CPU AND GPU
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
    
# FUNCION RMSE
def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))    
    
# ARRAY TO MATRIX
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)
    
    
    
    
print(get_available_devices())
    
# fix random seed for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(7)
np.random.seed(7)
tf.random.set_seed(7)


# DATASET LOADING
dataframe = pd.read_csv('D:/OneDrive - UVa/UVa/06 Sexto/TFG Teleco/Noleakage/7x7/Traffic_7x7_1.csv', usecols=range(1,50), engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
num_features = dataset.shape[1] #7x7=49


# BATCH NORMALIZATION
scaler = MinMaxScaler(feature_range=(0, 1))

# se escala con los datos de test. DATA LEAKAGE
dataset = scaler.fit_transform(dataset)
dataset = dataset[0:1475]
# SEPARATE INTO TRAIN AND TEST

train_size = 400#int(len(dataset) * 0.7)
test_size = 1075#len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:train_size+test_size, :]
print(len(train), len(test))




#train = scaler.fit_transform(train)
#test = scaler.transform(test)


# X=[T,T+LOOKBACK]  Y=[T+LOOCKBACK+1]
# look_back define cuántas muestras se utiliza para realizar una predicción
look_back = 24  #este parámetro, permite definir una ventana deslizante
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(testX.shape)
print(testY.shape[0])

# SET THE MODEL PARAMETERS

#batch_size indica cuántas ventanas look_back se procesan antes de actualizar los pesos (durante el entrenamiento)
# el numero total de muestras (entrenamiento) ha de ser divisible por batch_size 
batch_size = 25
# Cuántas vueltas al dataset entero se dan para entrenar
n_epochs = 400


# LSTM MODEL
myLSTM = Sequential()
myLSTM.add(LSTM(look_back, input_shape=(look_back, trainX.shape[2]), batch_input_shape=(batch_size, trainX.shape[1], trainX.shape[2]), stateful=True))
myLSTM.add(Dense(trainX.shape[2]))
myLSTM.compile(loss='mean_squared_error', optimizer='adam')
myLSTM.summary()




# TRAIN LSTM MODEL
#El reseteo del estado debe hacerse así cada epoch. Hacerlo es recomendable, si bien para datasets pequeños no es relevante
print('Fitting myLSTM on verbose 0')
for i in range(n_epochs):
    #Shuffle=True
    myLSTM.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0, shuffle=True)
    myLSTM.reset_states()

#El myLSTMos LSTM es stateful, es decir, que guarda sus estados h(t) y c(t) y los usa como valor inicial en el siguiente batch
#Como shuffle=True, el orden de los batches es aleatorio. Lo que significa que actúa como un stateless.





# STATIC TEST
train_Y_u = scaler.inverse_transform(trainY)
test_Y_u = scaler.inverse_transform(testY)

train_Yhat_s = myLSTM.predict(trainX, batch_size=batch_size)
train_Yhat_su = scaler.inverse_transform(train_Yhat_s)

test_Yhat_s = myLSTM.predict(testX, batch_size=batch_size)
test_Yhat_su = scaler.inverse_transform(test_Yhat_s)






# ONLINE LEARNING TEST
n_epochs_update = 5
error_threshold = 0
rmse_norm = 1
online_yhat_u = list()
times = list()
optimizer = tf.keras.optimizers.Adam()


# Initializethe LSTM with the last train batch (chronological) so the states are not zero
# Acts as recent memory context
online_X = trainX[-batch_size:,:,:]
online_Y = trainY[-batch_size:,:]

print('Applying SKIMA backpropagation')

# Online learning from the test set
# It updates the weights after processing 5 times the same batch (25 sequences of 24 hours)
for i in range(testY.shape[0]-batch_size):
    if rmse_norm > error_threshold:
        #print(f"RMSE NORM for sample {i}: {rmse_norm}")
        
        #Start timer to check computation speed
        st = tt.time()
        
        #Processing 5 times the same batch
        for j in range(n_epochs_update):
            with tf.GradientTape() as tape:
                # Get train yhat
                predictions = myLSTM(online_X, training=True)
                # Calculate rmse
                loss = rmse_loss(online_Y, predictions)
            
            # Get the error gradients
            gradients = tape.gradient(loss, myLSTM.trainable_variables)
            # Modifica los gradientes según rmse_norm
            adjusted_gradients = [g * (rmse_norm**2) for g in gradients]
            # Aplica los gradientes
            optimizer.apply_gradients(zip(adjusted_gradients, myLSTM.trainable_variables))
            # Borra los estados
            myLSTM.reset_states()
        
        #Pause timer
        et = tt.time()
        elapsed = et-st
        times.append(elapsed)
        
        
    # Get the next batch in chronological order
    newX, newY = testX[i:i+batch_size,:,:], testY[i:i+batch_size,:]
    
    # Predict the next batch
    online_yhat = myLSTM.predict(newX,batch_size=batch_size)
    online_yhat = scaler.inverse_transform(online_yhat)
    newY_u = scaler.inverse_transform(newY)
    
    # Get rmse of next batch
    rmse_norm =((np.sqrt(np.mean(((newY_u[-1,:] - online_yhat[-1,:])/online_yhat[-1,:])**2))))
    
    online_yhat_u.append(online_yhat[-1])
    
    # Mover la ventana del batch
    online_X = newX #Meto sólo un batch en el nuevo entrenamiento (no lo meneo mucho)
    online_Y = newY
    #Empezar de nuevo con el nuevo batch (procesar 5 veces, coger el batch siguiente, predecir sobre ese batch)











test_Yhat_ou=np.array(online_yhat_u)
times=np.array(times)


print(test_Y_u.shape)
print(test_Yhat_su.shape)
print(test_Yhat_ou.shape)
print(train_Y_u.shape)
print(train_Yhat_su.shape)
print(dataset.shape)


# calculate root mean squared error
trainScore=np.empty(num_features)
testScore=np.empty(num_features)
testScore_inc=np.empty(num_features)

for i in range(num_features):
    trainScore[i] = np.sqrt(mean_squared_error(train_Y_u[:,i], train_Yhat_su[:,i]))/np.std(train_Y_u[:,i])
    testScore[i] = np.sqrt(mean_squared_error(test_Y_u[:,i], test_Yhat_su[:,i]))/np.std(test_Y_u[:,i])
    testScore_inc[i] = np.sqrt(mean_squared_error(test_Y_u[batch_size-1:-1,i], test_Yhat_ou[:,i]))/np.std(test_Y_u[batch_size-1:-1,i])

timeScore = np.mean(times)




# Crear un DataFrame con los resultados
region_names = dataframe.columns.tolist()
results = {
    'Region' : region_names,
    'Train Score': trainScore,
    'Test Score': testScore,
    'Test Score incremental': testScore_inc,
    'Time Score': [timeScore] * num_features  # Repetir el valor de timeScore para todas las filas
}

df = pd.DataFrame(results)

combined_predictions = np.hstack((test_Y_u[batch_size-1:-1,:], test_Yhat_su[batch_size-1:-1, :], test_Yhat_ou))

all_cols = [f'Real_{name}' for name in region_names] + \
           [f'PredStatic_{name}' for name in region_names] + \
           [f'PredInc_{name}' for name in region_names]
           
df2 = pd.DataFrame(combined_predictions, columns=all_cols)

filename = f'resultados_norandom3_7x7LSTM_LB{look_back}_BS{batch_size}_EU{n_epochs_update}_T{error_threshold}_GWOL'
filenameexcel = f'{filename}.xlsx'
filenameplot = f'plot{filename}'

with pd.ExcelWriter(filenameexcel) as writer:
    df.to_excel(writer, sheet_name='Summary',index=False)
    df2.to_excel(writer, sheet_name='AllPredictions',index=False)
    


plt.figure(figsize=(15,6))

plt.plot(np.arange(num_features), trainScore, label='Train', marker='o', color='orange')
plt.plot(np.arange(num_features), testScore, label='StaticTest', marker='s', color='green')
plt.plot(np.arange(num_features), testScore_inc, label='OnlineTest', marker='^', color='firebrick')

plt.xticks(np.arange(num_features), region_names, rotation=90)
plt.tight_layout()

plt.xlabel('Region')
plt.ylabel('RNMSE')
plt.title(f'LSTM 7x7 {n_epochs} epochs {n_epochs_update} epoch update')
plt.legend(loc='best')

plt.savefig(filenameplot, dpi=300)

plt.show()