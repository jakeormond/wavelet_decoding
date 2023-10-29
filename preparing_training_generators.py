'''Preparing my data to use the deepinsight code
'''

import h5py
import numpy as np

# hard coded training inputs
# note we are not currently looking at speed


# first, need to get train and test times (splits)
# using 5 splits in total (5-fold cross validation)
# 4 splits for training, 1 for testing

def get_cv_splits(h5_path, n_splits=5, n_timesteps=64, batch_size=8):
    '''Get train and test times (splits) for cross validation
    '''

    hdf5_file = h5py.File(h5_path, mode='r')
    wave_data = hdf5_file['wave_mad_norm']

    exp_indices = np.arange(0, wave_data.shape[1] - n_timesteps * batch_size)
    cv_splits = np.array_split(exp_indices, n_splits)

    # for cv_run, cvs in enumerate(cv_splits):
    # K.clear_session()
    # For cv
    training_indices = np.setdiff1d(exp_indices, cv_splits[n_splits-1])  # All except the test indices
    testing_indices = cv_splits[n_splits-1]
    time_indices = {}
    time_indices['training'] = training_indices.tolist() 
    time_indices['testing'] = testing_indices.tolist() 

    return time_indices


def get_model_parameters(input_data_path, loss_functions, loss_weights):
    params = dict()
    params['input_data_path'] = input_data_path  # Filepath for hdf5 file storing wavelets and outputs
    params['sampling_rate'] = 30  # Sampling rate of the wavelets
    
    # -------- MODEL PARAMETERS --------------
    #params['model_function'] = 'the_decoder'  # Model architecture used
    params['model_timesteps'] = 64  # How many timesteps are used in the input layer, e.g. a sampling rate of 30 will yield 2.13s windows. Has to be divisible X times by 2. X='num_convs_tsr'
    params['num_convs_tsr'] = 4  # Number of downsampling steps within the model, e.g. with model_timesteps=64, it will downsample 64->32->16->8->4 and output 4 timesteps
    params['average_output'] = 2**params['num_convs_tsr']  # Whats the ratio between input and output shape
    # params['channel_lower_limit'] = 2

    params['optimizer'] = 'adam'  # Learning algorithm
    params['learning_rate'] = 0.0007  # Learning rate
    params['kernel_size'] = 3  # Kernel size for all convolutional layers
    params['conv_padding'] = 'same'  # Which padding should be used for the convolutional layers
    params['act_conv'] = 'elu'  # Activation function for convolutional layers
    params['act_fc'] = 'elu'  # Activation function for fully connected layers
    params['dropout_ratio'] = 0  # Dropout ratio for fully connected layers
    params['filter_size'] = 64  # Number of filters in convolutional layers
    params['num_units_dense'] = 1024  # Number of units in fully connected layer
    params['num_dense'] = 2  # Number of fully connected layers
    params['gaussian_noise'] = 1  # How much gaussian noise is added (unit = standard deviation)

    # -------- TRAINING----------------------
    params['batch_size'] = 8  # Batch size used for training the model
    params['steps_per_epoch'] = 250  # Number of steps per training epoch
    params['validation_steps'] = 250  # Number of steps per validation epoch
    params['epochs'] = 20  # Number of epochs
    params['shuffle'] = True  # If input should be shuffled
    params['random_batches'] = True  # If random batches in time are used
    params['metrics'] = []
    params['last_layer_activation_function'] = 'linear'
    params['handle_nan'] = False

    params['loss_functions'] = loss_functions
    params['loss_weights'] = loss_weights

    # -------- MISC--------------- ------------
    params['tensorboard_logfolder'] = './'  # Logfolder for tensorboard
    params['model_folder'] = './'  # Folder for saving the model
    params['log_output'] = False  # If output should be logged
    params['save_model'] = False  # If model should be saved

    return params

