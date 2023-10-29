"""
Original code from: 
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""

import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence

# from . import hdf5
import h5py
import hdf5_file_handling as hdf5


def create_train_and_test_generators(params, time_indices):
    """
    Creates training and test generators given opts dictionary

    Returns
    -------
    training_generator : object
        Sequence class used for generating training data
    testing_generator : object
        Sequence class used for generating testing data
    """
    # 1.) Create training generator
    training_generator = RawWaveletSequence(params, time_indices['training'], training=True)
    # 2.) Create testing generator
    testing_generator = RawWaveletSequence(params, time_indices['testing'], training=False)
    # 3.) Assert that training and testing data are different

    return (training_generator, testing_generator)


class RawWaveletSequence(Sequence):
    """
    Data Generator class. Import functions are get_input_sample and get_output_sample. 
    Each call to __getitem__ will yield a (input, output) pair

    Parameters
    ----------
    Sequence : object
        Keras sequence

    Yields
    -------
    input_sample : array_like
        Batched input for model training
    output_sample : array_like
        Batched output for model optimization

    --------------------




    """

    def __init__(self, params, time_indices, training=True): # training is a boolean to indicate whether this is a training or testing generator
        # 1.) Set all options as attributes
        self.set_params_as_attribute(params)

        # 2.) Load data memmaped for mean/std estimation and fast plotting
        self.wavelets = hdf5.read_hdf_memmapped(self.input_data_path, 'wave_mad_norm')

        # Get output(s)
        outputs = []
        for key, _ in params['loss_functions'].items():
            tmp_out = hdf5.read_hdf_memmapped(self.input_data_path, key)
            outputs.append(tmp_out)
        self.outputs = outputs

        # 3.) Prepare for training
        self.training = training
        self.prepare_data_generator(time_indices, training=training)

    def __len__(self):   # from tensor flow documentation, this is number of batches in the sequence
        # deepinsight code seems to set this as number of training examples, which I don't think is right
        return len(self.time_indices)  

    def __getitem__(self, idx):  # gets batch at position idx
        # 1.) Define start and end index

        # cut_range should be a 2d array of shape (batch_size, model_timesteps) containing 
        # the indices of the wavelets to be used for each batch.
        # each row is a training example - Jake

        if self.shuffle:
            idx = np.random.choice(self.time_indices)
        else:
            idx = self.time_indices[idx]
        cut_range = np.arange(idx, idx + self.sample_size) # sample size is length of 1 training example X batch size; is 64ts * 8training examples = 512

        # 2.) Above takes consecutive batches, implement random batching here
        if self.random_batches:
            indices = np.random.choice(self.time_indices, size=self.batch_size)
            cut_range = [np.arange(start_index, start_index + self.model_timesteps) for start_index in indices]
            cut_range = np.array(cut_range)
        else:
            cut_range = np.reshape(cut_range, (self.batch_size, cut_range.shape[0] // self.batch_size))

        # 3.) Get input sample
        input_sample = self.get_input_sample(cut_range)

        # 4.) Get output sample
        output_sample = self.get_output_sample(cut_range)

        return (input_sample, output_sample)

    def get_input_sample(self, cut_range):
        # 1.) Cut Ephys / fancy indexing for memmap is planned, if fixed use: cut_data = self.wavelets[cut_range, self.fourier_frequencies, self.channels]
        # cut_data = self.wavelets[cut_range, :, :] 
        # MY DATA IS A DIFFERENT SHAPE, IT IS FREQ X TIME X CHANNELS
        # THERE DATA WAS TIME X CHANNELS X FREQ (NEED TO VERIFY THIS!)
        # Because cut_range is batch_size x model_timestep, cut_data will be batch_size x model_timesteps x freq x channels

        cut_data = self.wavelets[:, cut_range, :] # my cut_data is freq x batch_size x model_timesteps x channels
        
        # cut_data = np.reshape(cut_data, (cut_data.shape[0] * cut_data.shape[1], cut_data.shape[2], cut_data.shape[3]))
        # this reshapes it to (batch_size * model_timesteps) x freq x channels

        cut_data = np.transpose(cut_data, axes=(1, 2, 0, 3)) # to make it batch_size x model_timesteps x freq x channels
        # I don't need to reshape becuase I've already normalized

        # 2.) Normalize input
        # mine is already normalized
        # cut_data = (cut_data - self.est_mean) / self.est_std

        # 3.) Reshape for model input--------- they reshape back to batch_size x model_timesteps x freq x channels
        # I don't need to do this
        # cut_data = np.reshape(cut_data, (self.batch_size, self.model_timesteps, cut_data.shape[1], cut_data.shape[2]))

        # 4.) Take care of optional settings
        cut_data = np.transpose(cut_data, axes=(0, 3, 1, 2)) # now it's batch_size x channels x model_timesteps x freq
        cut_data = cut_data[..., np.newaxis] # this adds a new axis at the end with size 1

        return cut_data

    def get_output_sample(self, cut_range):
        # 1.) Cut Ephys
        # outputs I'm using are position (dim time x 2) and head direction (dim time)
        out_sample = []
        for out in self.outputs:
            cut_data = out[cut_range, ...] # cut_range is batch_size x model_timesteps
            # if out is 1d, cut_data will be batch_size x model_timesteps
            # if out is 2d, cut_data will be batch_size x model_timesteps x 2...
            # and a training example will be training_example x model_timesteps x 2

            # then it is reshaped to (batch_size*model_timesteps) x 3rd dimension (but 3rd dimension shouldn't always exist???!!!)
            # seems unnecessary to me, it's already in the right shape
            # cut_data = np.reshape(cut_data, (cut_data.shape[0] * cut_data.shape[1], cut_data.shape[2]))

            # 2.) Reshape for model output
            # if len(cut_data.shape) is not self.batch_size: # this line must be wrong!!!
                # reshaped to batch_size x model_timesteps x 3rd dimension (if it exists!!!)
            # cut_data = np.reshape(cut_data, (self.batch_size, self.model_timesteps, cut_data.shape[1]))

            # 3.) Divide evenly and make sure last output is being decoded
            if self.average_output: # this must be downsampling the output (the factor is 16, from 64 ts to 4)
                # cut_data's index will be [:, [15, 31, 47, 63], :]
                cut_data = cut_data[:, np.arange(0, cut_data.shape[1] + 1, self.average_output)[1::] - 1] # 
            out_sample.append(cut_data)

        return out_sample

    def prepare_data_generator(self, time_indices, training):
        # 1.) Define sample size and means
        self.sample_size = self.model_timesteps * self.batch_size # each training example is 64 timesteps long, and 8 batches per step
        self.time_indices = time_indices
       
            
        # Make sure random choice takes from array not list 500x speedup
        self.time_indices = np.array(self.time_indices)
            
        # 9.) Calculate normalization for wavelets
        # meanstd_path = os.path.dirname(self.fp_hdf_out) + '/models/tmp/' + os.path.basename(self.fp_hdf_out)[:-3] + '_meanstd_start{}_end{}_tstart{}_tend{}.p'.format(
        #    self.training_indices[0], self.training_indices[-1], self.testing_indices[0], self.testing_indices[-1])
        
        # HAVE REMOVED MAD NORMALIZATION CODE BECAUSE I ALREADY RAN THE MAD NORMALIZATION

        # REMOVED NAN CODE BECAUSE THERE SHOULDN'T BE ANY NANs IN THE DATA
            
        # 10.) Define output shape. Most robust way is to get a dummy input and take that shape as output shape
        (dummy_input, _) = self.__getitem__(0)
        # Corresponds to the output of this generator, aka input to model. Also remove batch shape,
        self.input_shape = dummy_input.shape[1:] # this should be number of timesteps in 1 example

    def set_params_as_attribute(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def get_name(self):
        name = ""
        for attr in self.important_attributes:
            name += attr + ':{},'.format(getattr(self, attr))
        return name[:-1]
    
    