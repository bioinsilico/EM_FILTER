from tensorflow import keras as keras
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction import image

n = 240#16

model = keras.models.Sequential()

#2Dto1D
#model.add( keras.layers.Reshape((n*n,1),input_shape=(n,n,1)) )

#FC1
dim_net = 2*n*n
INPUT_SHAPE=(n,n,1)
input_img = keras.layers.Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoder = keras.layers.MaxPooling2D((2, 2), padding='same', name='encoder')(x)


x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoder = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.models.Model(input_img, decoder)
optimizer= keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.01, nesterov=False)
autoencoder.compile(loss='mse', optimizer=optimizer)

encoder_model = keras.models.Model(inputs=input_img, outputs=encoder)

model = autoencoder
sgd = keras.optimizers.RMSprop(lr=2e-5, rho=0.9, epsilon=1e-10, decay=0)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.summary()

X = np.load("../DATA_SETS/004773_ProtRelionRefine3D/images.numpy.npy")
#I = [image.extract_patches_2d(x, patch_size=(n,n), max_patches=10, random_state=1) for x in X]
#I = np.concatenate(I)
X = np.expand_dims(X,-1)

Y = np.load("../DATA_SETS/004773_ProtRelionRefine3D/projections.numpy.npy")
#I = [image.extract_patches_2d(x, patch_size=(n,n), max_patches=10, random_state=1) for x in Y]
#I = np.concatenate(I)
Y = np.expand_dims(Y,-1)

#(X,Y),(X_t,Y_t) = keras.datasets.mnist.load_data()
#
#X = X/255
#
#X = np.expand_dims(X,-1)
#Y= X

print(X.shape)
print(Y.shape)

dc3 = decoder
input_layer = input_img

tf.summary.image('output', dc3)
tf.summary.image('input',input_layer)
tf.summary.image('target',autoencoder.targets[0])

tfCback= keras.callbacks.TensorBoard(log_dir='../tb_tmp', histogram_freq=1, write_graph=True, write_grads=True, write_images=True)
#reduceLrCback= keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_lr=1e-6)
#model.fit(X, Y, validation_split=0.1, callbacks=[tfCback, reduceLrCback], batch_size=128, epochs=10)
model.fit(X, Y, validation_split=0.1, callbacks=[tfCback], batch_size=8, epochs=100)





