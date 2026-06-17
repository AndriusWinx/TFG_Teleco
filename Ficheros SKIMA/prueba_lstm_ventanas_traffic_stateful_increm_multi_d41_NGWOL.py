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
import time as tt


from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices())
# my output was => ['/device:CPU:0']
# good output must be => ['/device:CPU:0', '/device:GPU:0']

# fix random seed for reproducibility
tf.random.set_seed(7)


# load the dataset
dataframe = pd.read_csv('D:/OneDrive - UVa/UVa/06 Sexto/TFG Teleco/Ficheros SKIMA/Traffic3pixels.csv', usecols=[1,2,3], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
dataset=dataset[0:1400]
# split into train and test sets
train_size = 700#int(len(dataset) * 0.7)
test_size = 700#len(dataset) - train_size
#train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
train, test = dataset[0:train_size, :], dataset[train_size:train_size+test_size, :]
print(len(train), len(test))

# Definir la función de pérdida RMSE
def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
# Configurar el optimizador
optimizer = tf.keras.optimizers.Adam()



# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 24  #este parámetro, permite definir una ventana deslizante
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
#Así sería para expresar las features con el look_back
#trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Así sería la mejor representación, los time_steps, y el número de features en este caso igual a 1, porque sólo tenemos una medida, pero podemos tener varias variables observadas
# reshape input to be [samples, time steps, features]
#trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
#testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
n_epochs=400
batch_size=25
# create and fit the LSTM network
model = Sequential()
#model.add(LSTM(4, input_shape=(1, look_back))) #This changes for the "time_step" modelling

model.add(LSTM(look_back, input_shape=(look_back,trainX.shape[2]),batch_input_shape=(batch_size, trainX.shape[1], trainX.shape[2]),stateful=True))

#model.add(LSTM(24, input_shape=(look_back,1))) #Actually, second_dimension should be the number of features (in vector regression)

#model.add(tf.keras.layers.InputLayer(batch_input_shape=(batch_size,trainX.shape[1],trainX.shape[2])))
#model.add(LSTM(look_back,input_shape=(look_back,trainX.shape[2]),stateful=True))

model.add(Dense(trainX.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
n_epochs_update=5
# Error threshold
error_threshold =0
rmse_norm=1





#El reseteo del estado debe hacerse así cada epoch. Hacerlo es recomendable, si bien para datasets pequeños no es relevante
print('Fitting model on verbose 0')
for i in range(n_epochs):
 model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0, shuffle=True)
 model.reset_states()

# make predictions
trainPredict = model.predict(trainX,batch_size=batch_size)
testPredict = model.predict(testX,batch_size=batch_size)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainYground = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testYground = scaler.inverse_transform(testY)

trainX_copy=trainX[-batch_size:,:,:]
trainY_copy=trainY[-batch_size:,:]
predictions_score=list()
times=list()
print('Running ARIMA backpropagation')
for i in range(testY.shape[0]-batch_size):
    if rmse_norm > error_threshold:
        #print(f"RMSE NORM for sample {i}: {rmse_norm}")
        st=tt.time()
        for j in range(n_epochs_update):
            with tf.GradientTape() as tape:
                predictions = model(trainX_copy, training=True)
                loss = rmse_loss(trainY_copy, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            # Modifica los gradientes según rmse_norm
            #
            #
            #adjusted_gradients = [g * (rmse_norm**2) for g in gradients]
            adjusted_gradients = [g * (1) for g in gradients]
            optimizer.apply_gradients(zip(adjusted_gradients, model.trainable_variables))
            model.reset_states()
        '''
        for j in range(n_epochs_update):
            model.fit(trainX_copy, trainY_copy, epochs=1, batch_size=batch_size, verbose=2, shuffle=True)
            # with tf.GradientTape() as tape:
            #     predictions = model(trainX_copy, training=True)
            #     loss = rmse_loss(trainY_copy, predictions)
            #gradients = tape.gradient(loss,model.trainable_variables)  # Modifica los gradientes según rmse_norm
            #adjusted_gradients = [g * (1 + rmse_norm) for g in gradients]
            #optimizer.apply_gradients(zip(adjusted_gradients, model.trainable_variables))
            model.reset_states()
        '''
        et=tt.time()
        elapsed=et-st
        times.append(elapsed)
    # predict
    X, y = testX[i:i+batch_size,:,:], testY[i:i+batch_size,:]
    yhat = model.predict(X,batch_size=batch_size)
    yhat = scaler.inverse_transform(yhat)
    actual = scaler.inverse_transform(y)
    rmse_norm =((np.sqrt(np.mean(((actual[-1,:] - yhat[-1,:])/yhat[-1,:])**2))))

    predictions_score.append(yhat[-1])
    if i==0:
        #trainX_copy = np.vstack((trainX_copy[batch_size:,:,:], X))
        #trainY_copy = np.vstack((trainY_copy[batch_size:,:], y))
        trainX_copy = X #Meto sólo un batch en el nuevo entrenamiento (no lo meneo mucho
        trainY_copy = y
    else:
        trainX_copy = np.vstack((trainX_copy[1:,:,:], X[-1,:,:].reshape(1,look_back,-1)))
        trainY_copy = np.vstack((trainY_copy[1:,:], np.reshape(y[-1,:],(1,3))))

predictions_score=np.array(predictions_score)
times=np.array(times)


# calculate root mean squared error
trainScore=np.empty(3)
testScore=np.empty(3)
testScore_inc=np.empty(3)
for i in range(testY.shape[1]):
    trainScore[i] = np.sqrt(mean_squared_error(trainYground[:,i], trainPredict[:,i]))/np.std(trainYground[:,i])
    print('Train Score: %.4f RNMSE' % (trainScore[i]))
    testScore[i] = np.sqrt(mean_squared_error(testYground[:,i], testPredict[:,i]))/np.std(testYground[:,i])
    print('Test Score: %.4f RNMSE' % (testScore[i]))
    testScore_inc[i] = np.sqrt(mean_squared_error(testYground[batch_size - 1:-1,i], predictions_score[:,i]))/np.std(testYground[batch_size - 1:-1,i])
    print('Test Score incremental: %.4f RNMSE' % (testScore_inc[i]))
    # Calculate average time elapsed in each incremental training
    timeScore = np.mean(times)
print('Average incremental training time: %.4f s' % (timeScore))



# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back-1:len(trainPredict)+look_back-1, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset)-2, :] = testPredict
testPredictPlotInc = np.empty_like(dataset)
testPredictPlotInc[:, :] = np.nan
testPredictPlotInc[len(trainPredict)+(look_back*2)+batch_size:len(dataset)-2, :] = predictions_score
dataplot=scaler.inverse_transform(dataset)
real_data_for_comparison=dataplot[train_size+(2*look_back)+1:-1,:]
testPredict_for_comparison=testPredict[look_back+1:,:]

# Crear un DataFrame con los resultados
results = {
    'Train Score': trainScore,
    'Test Score': testScore,
    'Test Score incremental': testScore_inc,
    'Time Score': [timeScore] * len(trainScore)  # Repetir el valor de timeScore para todas las filas
}

df = pd.DataFrame(results)



# Guardar el DataFrame en un archivo Excel
filename = f'resultados{train_size}_{n_epochs}_{n_epochs_update}_{error_threshold}_NGWOL.xlsx'
df.to_excel(filename, index=False)

## Combine arrays to compare horizontally (axis=1)
combined_array = np.hstack((real_data_for_comparison,testPredict_for_comparison, predictions_score))
# Convert the combined array to a pandas DataFrame
df = pd.DataFrame(combined_array, columns=['Real1', 'Real2', 'Real3','Pred1', 'Pred2', 'Pred3', 'PredInc1', 'PredInc2', 'PredInc3'])
# Save the DataFrame to an Excel file

filename = f'resultados_for_plot{train_size}_{n_epochs}_{n_epochs_update}_{error_threshold}_NGWOL.xlsx'
df.to_excel(filename, index=False)
print(f'Resultados guardados en {filename}')

labels=["RealData","Predicted training data","Predicted test data, no online learning","NGWOL"]

fig,ax=plt.subplots(3,sharex=True)

for i in range(3):
    # plot baseline and predictions
    ax[i].plot(dataplot[:,i],label=labels[0])
    ax[i].plot(trainPredictPlot[:,i],label=labels[1])
    ax[i].plot(testPredictPlot[:,i],label=labels[2])
    ax[i].plot(testPredictPlotInc[:,i],label=labels[3])

# Create a global legend below all subplots
fig.legend(labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.15))
# Label for the X-axis
ax[0].set_ylabel("Region 4259")
ax[1].set_ylabel("Region 4456")
ax[2].set_ylabel("Region 5060")
ax[-1].set_xlabel("Time (h)")
# Adjust layout to fit the legend
fig.tight_layout()
fig.subplots_adjust(bottom=0.1)
# Add space for the global legend


# Guardar la figura con un nombre que dependa de a y b
filename2 = f'resultados_{train_size}_{n_epochs}_{n_epochs_update}_{error_threshold}_NGWOL.png'  # Genera un nombre dinámico
plt.savefig(filename2, dpi=300, bbox_inches='tight')  # Guarda la figura con alta calidad
plt.show()
print(f"Figura guardada como {filename2}")


