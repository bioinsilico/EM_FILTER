from tensorflow import keras as keras
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction import image

n = 64 

model = keras.models.Sequential()

input_dim = n
dim_net = int(n*n)

#FC1
input_layer = keras.layers.InputLayer(input_shape=(input_dim,input_dim,1))
model.add( input_layer )

#FC2
model.add(keras.layers.Dense(units=dim_net, activation='tanh'))

#FC3
model.add(keras.layers.Dense(units=dim_net, activation='tanh'))

#C1
#model.add( keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu',input_shape=(n,n,1) )) 
model.add( keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu') )
#model.add( keras.layers.LeakyReLU(alpha=0.2) )

#C2
#model.add( keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu', activity_regularizer=keras.regularizers.l1(0.0001) ))
model.add( keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu' ))
#model.add( keras.layers.LeakyReLU(alpha=0.2) )

#DC3
dc3 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(7,7), strides=(1, 1), padding='same', activation='linear')
model.add( dc3 )


sgd = keras.optimizers.RMSprop(lr=2e-5, rho=0.9, epsilon=1e-10, decay=0)
#sgd = keras.optimizers.SGD(lr=2e-2, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.summary()

X = np.load("../DATA_SETS/004773_ProtRelionRefine3D/images.numpy.npy")
I = [image.extract_patches_2d(x, patch_size=(n,n), max_patches=10, random_state=1) for x in X]
I = np.concatenate(I)
X = np.expand_dims(I,-1)

Y = np.load("../DATA_SETS/004773_ProtRelionRefine3D/projections.numpy.npy")
I = [image.extract_patches_2d(x, patch_size=(n,n), max_patches=10, random_state=1) for x in Y]
I = np.concatenate(I)
Y = np.expand_dims(I,-1)

print(X.shape)
print(Y.shape)

tf.summary.image('output', dc3.output)
tf.summary.image('input',input_layer.output)
tf.summary.image('target',model.targets[0])

tfCback= keras.callbacks.TensorBoard(log_dir='../tb_tmp', histogram_freq=1, write_graph=True, write_grads=True, write_images=True)
#reduceLrCback= keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_lr=1e-6)
#model.fit(X, Y, validation_split=0.1, callbacks=[tfCback, reduceLrCback], batch_size=128, epochs=10)
model.fit(X, Y, validation_split=0.1, callbacks=[tfCback], batch_size=8, epochs=300)





