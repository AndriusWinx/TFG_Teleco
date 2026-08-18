import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python.client import device_lib
import visualkeras
import time as tt


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
tf.random.set_seed(7)


# DATASET LOADING
dataframe = pd.read_csv('D:/OneDrive - UVa/UVa/06 Sexto/TFG Teleco/Ficheros SKIMA/Traffic3pixels.csv', usecols=[1,2,3], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')


# SEPARATE INTO TRAIN AND TEST
dataset = dataset[0:1400]
train_size = 400#int(len(dataset) * 0.7)
test_size = 1000#len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:train_size+test_size, :]
print(len(train), len(test))


# BATCH NORMALIZATION
scaler = MinMaxScaler(feature_range=(0, 1))
# se escala con los datos de test. DATA LEAKAGE
train = scaler.fit_transform(train)
test = scaler.transform(test)


# X=[T,T+LOOKBACK]  Y=[T+LOOCKBACK+1]
# look_back define cuántas muestras se utiliza para realizar una predicción
look_back = 24  #este parámetro, permite definir una ventana deslizante
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


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

myLSTM.save("pruebaimage3x1.h5")

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

print(test_Yhat_ou[:,0].shape)
print(test_Y_u[look_back:,0].shape)
print(test_Yhat_su.shape)
print(train_Y_u.shape)
print(train_Yhat_su.shape)



# calculate root mean squared error
trainScore=np.empty(3)
testScore=np.empty(3)
testScore_inc=np.empty(3)

for i in range(testY.shape[1]):
    trainScore[i] = np.sqrt(mean_squared_error(train_Y_u[:,i], train_Yhat_su[:,i]))/np.std(train_Y_u[:,i])
    print('Train Score: %.4f RNMSE' % (trainScore[i]))
    testScore[i] = np.sqrt(mean_squared_error(test_Y_u[:,i], test_Yhat_su[:,i]))/np.std(test_Y_u[:,i])
    print('Test Score: %.4f RNMSE' % (testScore[i]))
    testScore_inc[i] = np.sqrt(mean_squared_error(test_Y_u[look_back:-1,i], test_Yhat_ou[:,i]))/np.std(test_Y_u[look_back:-1,i])
    print('Test Score incremental: %.4f RNMSE' % (testScore_inc[i]))
    # Calculate average time elapsed in each incremental training
    timeScore = np.mean(times)
print('Average incremental training time: %.4f s' % (timeScore))



# shift train predictions for plotting
train_Yhat_staticPlot = np.empty_like(dataset)
train_Yhat_staticPlot[:, :] = np.nan
train_Yhat_staticPlot[look_back-1:len(train_Yhat_su)+look_back-1, :] = train_Yhat_su
# shift test predictions for plotting
test_yhat_staticPlot = np.empty_like(dataset)
test_yhat_staticPlot[:, :] = np.nan
test_yhat_staticPlot[len(train_Yhat_su)+(look_back*2):len(dataset)-2, :] = test_Yhat_su
test_yhat_staticPlotInc = np.empty_like(dataset)
test_yhat_staticPlotInc[:, :] = np.nan
test_yhat_staticPlotInc[len(train_Yhat_su)+(look_back*2)+batch_size:len(dataset)-2, :] = test_Yhat_ou
dataplot=dataset
real_data_for_comparison=dataplot[train_size+(2*look_back)+1:-1,:]
test_yhat_static_for_comparison=test_Yhat_su[look_back+1:,:]


# Crear un DataFrame con los resultados
results = {
    'Train Score': trainScore,
    'Test Score': testScore,
    'Test Score incremental': testScore_inc,
    'Time Score': [timeScore] * len(trainScore)  # Repetir el valor de timeScore para todas las filas
}

df = pd.DataFrame(results)



# Guardar el DataFrame en un archivo Excel
filename = f'resultados{train_size}_{n_epochs}_{n_epochs_update}_{error_threshold}_noleakage_GWOL.xlsx'
df.to_excel(filename, index=False)
print(real_data_for_comparison.shape)
print(test_yhat_static_for_comparison)
print(test_Yhat_ou)
# Combine arrays to compare horizontally (axis=1)
combined_array = np.hstack((real_data_for_comparison,test_yhat_static_for_comparison, test_Yhat_ou))
# Convert the combined array to a pandas DataFrame
df = pd.DataFrame(combined_array, columns=['Real1', 'Real2', 'Real3','Pred1', 'Pred2', 'Pred3', 'PredInc1', 'PredInc2', 'PredInc3'])
# Save the DataFrame to an Excel file

filename = f'resultados_for_plot_{train_size}_{n_epochs}_{n_epochs_update}_{error_threshold}_noleakage_GWOL.xlsx'
df.to_excel(filename, index=False)
print(f'Resultados guardados en {filename}')

labels=["RealData","Predicted training data","Predicted test data static","GWOL (our method)"]

fig,ax=plt.subplots(3,sharex=True)

for i in range(3):
    # plot baseline and predictions
    ax[i].plot(dataplot[:,i],label=labels[0])
    ax[i].plot(train_Yhat_staticPlot[:,i],label=labels[1])
    ax[i].plot(test_yhat_staticPlot[:,i],label=labels[2])
    ax[i].plot(test_yhat_staticPlotInc[:,i],label=labels[3])

# Create a global legend below all subplots
fig.legend(labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.15))
# Label for the X-axis
ax[0].set_ylabel("Region 4259")
ax[1].set_ylabel("Region 4456")
ax[2].set_ylabel("Region 5060")
ax[-1].set_xlabel("Time (h)")
# Adjust layout to fit the legend
fig.tight_layout()
#fig.subplots_ae global legend
fig.subplots_adjust(bottom=0.1)
# Add space for th

# Guardar la figura con un nombre que dependa de a y b
filename2 = f'resultados_{train_size}_{n_epochs}_{n_epochs_update}_{error_threshold}_noleakage_GWOL.png'  # Genera un nombre dinámico
plt.savefig(filename2, dpi=300, bbox_inches='tight')  # Guarda la figura con alta calidad
plt.show()
print(f"Figura guardada como {filename2}")


