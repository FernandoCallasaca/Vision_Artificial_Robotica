import sys
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
#from tensorflow.keras import adam
#tf.keras.optimizers.Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
K.clear_session()
data_entrenamiento="D:/data/entrenamiento/"
data_validacion="D:/data/validacion/"

#parametros
epocas=50
altura, longitud = 100, 100
batch_size=200
pasos=20
pasos_validacion=5
filtrosConv1=16
filtrosConv2=32
filtrosConv3=64
filtrosConv4=128
filtrosConv5=256
filtrosConv6=128
tamano_filtro1=(3,3)
tamano_filtro2=(3,3)
tamano_filtro3=(3,3)
tamano_filtro4=(3,3)
tamano_filtro5=(3,3)
tamano_filtro6=(2,2)
tamano_pool=(2,2)
clases=3
#lr=0.5
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

#preprocesamiento de imagenes
entrenamiento_datagen=ImageDataGenerator(  
    shear_range=0.3, 
    zoom_range=0.3, 
    horizontal_flip=True, 
    rescale=1./255
)
validacion_datagen=ImageDataGenerator(
    rescale=1./255
)
imagen_entrenamiento=entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical'
)
imagen_validacion=validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

#Red CNN
cnn=Sequential()
cnn.add(Convolution2D(filtrosConv1,tamano_filtro1,padding='same',input_shape=(altura,longitud,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv2,tamano_filtro2,padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv3,tamano_filtro3,padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv4,tamano_filtro4,padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv5,tamano_filtro5,padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv6,tamano_filtro6,padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.2))
cnn.add(Dense(clases,activation='softmax'))
#model.compile(optimizer=tf.keras.optimizers.Blah() metrics=[tf.keras.metrics.BinaryAccuracy()])
#cnn.compile(loss='categorical_crossentropy' optimizer=optimizers.adam(lr=lr) metrics= 'accuracy' )
#cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
cnn.fit(imagen_entrenamiento,steps_per_epoch=pasos,epochs=epocas,validation_data=imagen_validacion,validation_steps=pasos_validacion)
dir="D:/modelo/"
cnn.save("D:/modelo/modelo.h5")
cnn.save_weights("D:/modelo/pesos.h5")
