from tensorflow import keras as keras
import numpy as np

n = 28 

model = keras.models.Sequential()

##2Dto1D
#model.add( keras.layers.Reshape((n*n,1),input_shape=(n,n,1)) )
#
##FC1
#dim_net = 2*n*n
#model.add(keras.layers.Dense(units=dim_net, activation='tanh'))
#
##FC2
#dim_net = n*n
#model.add(keras.layers.Dense(units=dim_net, activation='tanh'))
#
##FC3
#dim_net = n*n
#model.add(keras.layers.Dense(units=dim_net, activation='tanh'))
#
##1Dto2D
#model.add( keras.layers.Reshape((n,n,1)) )

#C1
model.add( keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu',input_shape=(n,n,1)) )

#C2
model.add( keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu', activity_regularizer=keras.regularizers.l1(0.0001) ))

#DC3
model.add( keras.layers.Conv2DTranspose(filters=1, kernel_size=(7,7), strides=(1, 1), padding='same', activation='linear') )


sgd = keras.optimizers.RMSprop(lr=2e-5, rho=0.9, epsilon=1e-10, decay=0)

model.compile(loss='mean_squared_error', optimizer=sgd)

model.summary()

(X,Y),(X_t,Y_t) = keras.datasets.mnist.load_data()

X = X/255

X = np.expand_dims(X,-1)[:50, ...]
Y= X

print(X.shape)

tfCback= keras.callbacks.TensorBoard(log_dir='../tb_tmp', histogram_freq=1, write_graph=True, write_grads=True, write_images=True)
#reduceLrCback= keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_lr=1e-6)
#model.fit(X, Y, validation_split=0.1, callbacks=[tfCback, reduceLrCback], batch_size=128, epochs=10)
model.fit(X, Y, validation_split=0.1, callbacks=[tfCback], batch_size=8, epochs=3)

