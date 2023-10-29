'''Script for running the whole process, starting from the normalized
wavelet data (which was previously generated) and positional data, 
which is used to create the generators, then ...'''

from preparing_training_generators import get_cv_splits, get_model_parameters
from training_generator import create_train_and_test_generators

input_data_path = "/media/jake/LaCie/wave_data_05_05_2023/wave_data_contiguous.h5"

time_indices = get_cv_splits(input_data_path, n_splits=5, \
                             n_timesteps=64, batch_size=8)

# hard coded training inputs
# note we are not currently looking at speed
loss_functions = {'position' : 'euclidean_loss', 
                  'hd' : 'cyclical_mae_rad'}

loss_weights = {'position' : 1, 
                'hd' : 25}

params = get_model_parameters(input_data_path, loss_functions, loss_weights)

## make some directories
## and a tensorboard log folder (???)

# create generators
training_generator, testing_generator = \
    create_train_and_test_generators(params, time_indices)

pass
# train the model
