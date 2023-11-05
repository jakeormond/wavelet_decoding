'''Script for running the whole process, starting from the normalized
wavelet data (which was previously generated) and positional data, 
which is used to create the generators, then ...'''

from preparing_training_generators import get_cv_splits, get_model_parameters
from training_generator import create_train_and_test_generators
import model_architecture
from tensorflow import optimizers
import custom_losses 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import os
import datetime
import tensorflow as tf 

input_data_path = "/media/jake/LaCie/wave_data_05_05_2023/wave_data_contiguous_no_nan.h5"
input_data_dir = os.path.dirname(input_data_path)

time_indices = get_cv_splits(input_data_path, n_splits=5, \
                             n_timesteps=64, batch_size=8)

# loss functions
euclidean_loss = custom_losses.euclidean_loss
cyclical_mae_rad = custom_losses.cyclical_mae_rad

loss_functions = {'position' : euclidean_loss, 
                  'hd' : cyclical_mae_rad}

loss_weights = {'position' : 1, 
                'hd' : 25}

loss_metrics = []

# model parameters
params = get_model_parameters(input_data_path, loss_functions, loss_weights)

## make some directories
## and a tensorboard log folder (???)

# create generators
training_generator, testing_generator = \
    create_train_and_test_generators(params, time_indices)

# create model
with tf.keras.utils.custom_object_scope(loss_functions):
    model = model_architecture.the_decoder(training_generator, show_summary=True)

    # compile model
    opt = optimizers.Adam(learning_rate=training_generator.learning_rate, amsgrad=True)

    model.compile(loss=loss_functions, optimizer=opt, loss_weights=loss_weights, \
              metrics=loss_metrics)

# callbacks
# get date and time
date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
model_name = "deep_insight_" + date_time
file_name = model_name + '.hdf5'

model_dir = os.path.join(input_data_dir, 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

file_path = os.path.join(model_dir, file_name)
model_cp = ModelCheckpoint(filepath=file_path, save_best_only=True, save_weights_only=True)

reduce_lr_cp = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

tensorboard_logfolder = os.path.join(input_data_dir, 'tensorboard_logs', model_name)
tensorboard_cp = TensorBoard(log_dir=tensorboard_logfolder)

callbacks = [model_cp, reduce_lr_cp, tensorboard_cp]

# train model
history = model.fit(training_generator, steps_per_epoch=300, epochs=10, \
                    shuffle=training_generator.shuffle, validation_steps=300, \
                        validation_data=testing_generator, verbose=1, callbacks=callbacks)

pass
