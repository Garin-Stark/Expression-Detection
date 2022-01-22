from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import os

def count_files(path):
    return sum([len(files) for r, d, files in os.walk(path)])

train_data_dir = 'data/train'
validation_data_dir = 'data/test'

num_classes = len(os.listdir(train_data_dir))
img_rows,img_cols = 48,48
batch_size = 32
epochs = 50

train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


model = Sequential()

# Block-1

model.add(tf.keras.layers.Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))

# Block-2 

model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))

# Block-3

model.add(tf.keras.layers.Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))

# Block-4 

model.add(tf.keras.layers.Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))

# Block-5

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,kernel_initializer='he_normal'))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))

# Block-6

model.add(tf.keras.layers.Dense(64,kernel_initializer='he_normal'))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))

# Block-7

model.add(tf.keras.layers.Dense(num_classes,kernel_initializer='he_normal'))
model.add(tf.keras.layers.Activation('softmax'))

print(model.summary())

checkpoint = tf.keras.callbacks.ModelCheckpoint('trained_model.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

train_samples = count_files(train_data_dir)
validation_samples = count_files(validation_data_dir)

history=model.fit(
                train_generator,
                steps_per_epoch=train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=validation_samples//batch_size)