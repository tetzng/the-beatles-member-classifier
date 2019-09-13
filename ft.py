#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

classes = ['john_a', 'john_b', 'paul_young', 'paul_old', 'george_young', 'george_old', 'ringo_young', 'ringo_old']
nb_classes = len(classes)

img_width, img_height = 200, 200

train_data_dir = './data/train/'
validation_data_dir = './data/test/'

nb_train_samples = 400
nb_validation_samples = 400

batch_size = 16
nb_epoch = 20

train_datagen = ImageDataGenerator(
  rescale=1.0 / 255,
  zoom_range=0.2,
  horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=batch_size,
  shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=batch_size,
  shuffle=True)

input_tensor = Input(shape=(img_width, img_height, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))

vgg_model = Model(input=vgg16.input, output=top_model(vgg16.output))

for layer in vgg_model.layers[:15]:
    layer.trainable = False

vgg_model.compile(loss='categorical_crossentropy',
          optimizer=SGD(lr=1e-4, momentum=0.9),
          metrics=['accuracy'])

history = vgg_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

vgg_model.save_weights('vgg16_finetuning_weights.h5')

vgg_model.save('vgg16_finetuning_model.h5')
