import tensorflow as tf
import tempfile
import numpy as np
import datetime
%load_ext tensorboard

mnist = tf.keras.datasets.mnist # The MNIST dataset.

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

%reload_ext tensorboard

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='layers_flatten'),
    tf.keras.layers.Dense(512, activation='relu', name='layers_dense'),
    tf.keras.layers.Dropout(0.2, name='layers_dropout'),
    tf.keras.layers.Dense(10, activation='softmax', name='layers_dense_2')
  ])

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              steps_per_execution=10)

log_dir = tempfile.mkdtemp()
tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir=log_dir,
  histogram_freq=1) # Enable histogram computation with each epoch.

model.fit(x=x_train,
          y=y_train,
          epochs=10,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

%tensorboard --logdir {tensorboard_callback.log_dir}