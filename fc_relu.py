from tensorflow import keras as keras
import numpy as np

n = 240

model = keras.models.Sequential()

#2Dto1D
model.add( keras.layers.Reshape((n*n,1),input_shape=(n,n,1)) )

#FC1
dim_net = 2*n*n
model.add(keras.layers.Dense(units=dim_net, activation='relu'))

#OUT
dim_net = n*n
model.add(keras.layers.Dense(units=dim_net, activation='linear'))

#2Dto1D
model.add( keras.layers.Reshape((n,n,1)) )

sgd = keras.optimizers.RMSprop(lr=2e-5, rho=0.9, epsilon=10e-10, decay=0)

model.compile(loss='mean_squared_error', optimizer=sgd)

model.summary()

X = np.load("../DATA_SETS/004773_ProtRelionRefine3D/images.numpy.npy")
x = (X-X.min())/(X.max()-X.min())
X = np.expand_dims(X,-1)

Y = np.load("../DATA_SETS/004773_ProtRelionRefine3D/projections.numpy.npy")
Y = (Y-Y.min())/(Y.max()-Y.min())
Y = np.expand_dims(Y,-1)

print(X.shape)
print( np.isnan(X).any() )

print(Y.shape)
print( np.isnan(Y).any() )

tfCback= keras.callbacks.TensorBoard(log_dir='../tb_tmp', histogram_freq=1, write_graph=True, write_grads=True, write_images=True)
#reduceLrCback= keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_lr=1e-6)
#model.fit(X, Y, validation_split=0.1, callbacks=[tfCback, reduceLrCback], batch_size=128, epochs=10)
model.fit(X, Y, validation_split=0.1, callbacks=[tfCback], batch_size=64, epochs=100)

