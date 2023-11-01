explanation of network architecture 

In Markus's paper, they recorded from 32 tetrodes, giving a total of 128 channels. 

They computed wavelets at 26 scales across the 128 channels, and then downsampled
(or averaged???) them by 1000 (the sampling rate of the data was 30,000; this is 
the same as ours).

As input to the model, they used a 3d tensor spanning 64 times points, giving a 
shape of (64, 26, 128).
    - The order of these dimension wrong. 128 should be first dimension (after the batch dimension) for the timedistributed layer to work (because we initially use channels as the time dimension).

They then applied 2d convolutions across all the channels, dowsampling by 2 the
frequency dimension and time dimension on alternating layers using strides of 
(1,2) and (2,1) respectively in the first 8 layers, resulting in a tensor of
shape (4, 2, 128).

A lamda layer is used to permute the dimensions. 

They then applied 2d convolutions across all time points, dowsampling by 2 the 
channel dimension for next 6 layers, resulting in a tensor of shape (4, 2, 2).

