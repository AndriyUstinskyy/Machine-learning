# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

clf = Sequential()

#convolucion
clf.add(Conv2D(filters= 32, kernel_size = (3,3), input_shape = (64,64,3) ,
             activation = "relu"))

#max-pooling

clf.add(MaxPooling2D(pool_size = (2,2)))

clf.add(Flatten())

clf.add(Dense(output_dim = 128, activation = "relu"))
clf.add(Dense(output_dim = 1, activation = "sigmoid"))

clf.compile(optimizer = "adam", loss = "binary_crossentropy" , metrics = ["accuracy"])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( 
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

clf.fit_generator(training_dataset,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=testing_dataset,
                        validation_steps=2000)

