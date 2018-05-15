from tensorflow import keras as keras
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction import image
import sys

n = 128

model = keras.models.Sequential()

input_dim = n
input_conv_dim = int(0.5*n)

dim_net_FC2 = int(0.25*n*n)
dim_net_FC3 = int(0.25*n*n)

#FC1 - INPUT
input_layer = keras.layers.InputLayer(input_shape=(input_dim,input_dim,1))
model.add( input_layer )
model.add(keras.layers.Reshape( (input_dim*input_dim,1) ))
model.add(keras.layers.Flatten())

#FC2
model.add(keras.layers.Dense(units=dim_net_FC2, activation='tanh'))

#FC3
model.add(keras.layers.Dense(units=dim_net_FC3, activation='tanh'))

model.add(keras.layers.Reshape( (input_conv_dim,input_conv_dim,1) ))
#C1
model.add( keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu') )

#C2
model.add( keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu', activity_regularizer=keras.regularizers.l1(1e-5) ))

#DC3
dc3 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(7,7), strides=(1, 1), padding='same', activation='linear')
model.add( dc3 )

sgd = keras.optimizers.RMSprop(lr=2e-5, rho=0.9, epsilon=1e-10, decay=0)
#sgd = keras.optimizers.SGD(lr=2e-2, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.summary()

X = np.load("../DATA_SETS/004773_ProtRelionRefine3D/images.numpy.npy")
N_rand = X.shape[0]
R = np.random.randint(100,size=N_rand)

I = [image.extract_patches_2d(x, patch_size=(n,n), max_patches=10, random_state=r) for x,r  in zip(X,R)]
I = np.concatenate(I)
X = np.expand_dims(I,-1)

n1 = int(n*0.25)
n2 = int(n*0.75)
Y = np.load("../DATA_SETS/004773_ProtRelionRefine3D/projections.numpy.npy")
I_ = [image.extract_patches_2d(x, patch_size=(n,n), max_patches=10, random_state=r) for x,r in zip(Y,R)]
I_ = np.concatenate(I_)
I = [ x[n1:n2,n1:n2] for x in I_]
Y = np.array(I)
Y = np.expand_dims(I,-1)

FM = np.load("../DATA_SETS/004773_ProtRelionRefine3D/fraction_micrograph.numpy.npy")
FM = np.expand_dims(FM,-1)

print(X.shape)
print(Y.shape)
print(FM.shape)

tf.summary.image('output', dc3.output)
tf.summary.image('input',input_layer.output)
tf.summary.image('target',model.targets[0])

tfCback= keras.callbacks.TensorBoard(log_dir='../tb_tmp', histogram_freq=1, write_graph=True, write_grads=True, write_images=True)

class _predict_and_save(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    P = model.predict(FM)
    np.save("../DATA_SETS/004773_ProtRelionRefine3D/fraction_micrograph.predictions.numpy",P)
    return 

predict_and_save = _predict_and_save()
#reduceLrCback= keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_lr=1e-6)
#model.fit(X, Y, validation_split=0.1, callbacks=[tfCback, reduceLrCback], batch_size=128, epochs=10)
model.fit(X, Y, validation_split=0.1, callbacks=[tfCback,predict_and_save], batch_size=8, epochs=300)

