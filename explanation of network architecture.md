explanation of network architecture 

In Markus's paper, they recorded from 32 tetrodes, giving a total of 128 channels. 

They computed wavelets at 26 scales across the 128 channels, and then downsampled
(or averaged???) them by 1000 (the sampling rate of the data was 30,000; this is 
the same as ours).

As input to the model, they used a 3d tensor spanning 64 times points, giving a 
shape of (64, 26, 128).
    - The order of these dimension wrong. 128 should be first dimension (after the batch dimension) for the timedistributed layer to work (because we initially use channels as the time dimension).
    - In fact, my input data is in the right shape (385 ch x 64 tp x 25 freq x 1)

They then applied 2d convolutions across all the channels, dowsampling by 2 the
frequency dimension and time dimension on alternating layers using strides of 
(1,2) and (2,1) respectively in the first 8 layers, resulting in a tensor of
shape (None, 4, 2, 128; note that the shape method returns None for the batch dimension).
    -according to the documentation for TimeDistributed, the output shape should
    be (None, timesteps, output_dim, n_filters). output_dim is time x freq (so final dim is 4x2).


A lamda layer is used to permute the dimensions, so that they become batch x time x freq x channels x filters (i.e. 0, 2, 3, 1, 4). Becomes None x 4 x 2 x 128 x 64.

Then 2d convolutions across all time points using 128 filters, and a 1 x 2 kernel, dowsampling by 2 the channel dimension for next 6 layers, resulting in a tensor of shape (None, 4, 2, 2, 128).

Next, TimeDistributed(Flatten) to produce shape (None, 4, 512).

Then 2 loops of Dense(n=1024) and Dropout. Oddly, the dropout_ratio is 0, so the 
dropout layer will have no effect.  Shape is None, 4, 1024

Finally, a Dense layer with 1 or 2 output units (depending on whether output is 1d or 2d) and a linear activation, producing shape None, 4, 1 or 2.



