import h5py
import numpy as np

input_data_path = "/media/jake/LaCie/wave_data_05_05_2023/wave_data_new_concat.h5"
output_data_path = "/media/jake/LaCie/wave_data_05_05_2023/wave_data_contiguous_no_nan.h5"

# datasets to copy
ds_names = ['position', 'hd']

# Open the existing HDF5 file in read mode
nan_indices = {}
with h5py.File(input_data_path, 'r') as f:
    for ds_name in ds_names:
        # Open the existing chunked dataset
        chunked_dataset = f[ds_name]
        if len(chunked_dataset.shape) == 2:
            chunked_dataset = chunked_dataset[:, 0]

        # Get the shape and datatype of the existing dataset
        shape = chunked_dataset.shape
        dtype = chunked_dataset.dtype

        # find nans
        # Create a masked array with the same data
        masked_array = np.ma.masked_invalid(chunked_dataset)

        # Get the indices of NaN values
        nan_indices[ds_name] = np.argwhere(masked_array.mask)

# check that nan_indices['position'] and nan_indices['hd'] are the same
if np.array_equal(nan_indices['position'], nan_indices['hd']):
    print("nan_indices['position'] and nan_indices['hd'] are the same")

nan_indices = nan_indices['position']

# Open the existing HDF5 file in read mode
# ds_names = ['wave_mad_norm', 'position', 'hd']
ds_names = ['position', 'hd']
with h5py.File(input_data_path, 'r') as f:   

    for ds_name in ds_names:
        # Open the existing chunked dataset
        chunked_dataset = f[ds_name]
        shape = chunked_dataset.shape
        print(f"shape of {ds_name} before nan removal: {shape}")
       
        dtype = chunked_dataset.dtype
        
        # Create a new contiguous dataset with the same shape and datatype
        with h5py.File(output_data_path, 'a') as new_f:
            # check if dataset already exists
            if ds_name in new_f:
                print(f"dataset {ds_name} already exists in {output_data_path}")
                # open dataset
                new_dataset = new_f[ds_name]
            else:
                print(f"creating dataset {ds_name} in {output_data_path}")

                if ds_name == 'wave_mad_norm':
                    new_shape= (shape[0], shape[1] - nan_indices.shape[0], shape[2])
                elif ds_name == 'position':
                    new_shape = (shape[0] - nan_indices.shape[0], shape[1])
                elif ds_name == 'hd':
                    new_shape = (shape[0] - nan_indices.shape[0])

                new_dataset = new_f.create_dataset(ds_name, new_shape, dtype=dtype)

            if ds_name == 'wave_mad_norm':
                          # Copy data from the chunked dataset to the new contiguous dataset
                for i in range(new_shape[0]):
                    # Read a chunk from the chunked dataset
                    chunk = chunked_dataset[i, :, :]
                    chunk_shape = chunk.shape
                    print(f"shape of chunk {i} before nan removal: {chunk_shape}")
                    # remove nans from chunk
                    chunk = np.delete(chunk, nan_indices, axis=0)
                    print(f"shape of chunk {i} after nan removal: {chunk.shape}")

                    # Write the chunk to the new dataset at the corresponding location
                    new_dataset[i, :, :] = chunk
                    print(f"saved chunk {i}")

            else:
                # remove nans from chunked dataset
                chunked_dataset = np.delete(chunked_dataset, nan_indices, axis=0)                
                new_dataset[...] = chunked_dataset

            if new_dataset.chunks is None:
                print("new_dataset.chunks is None")
            else:
                print("new_dataset.chunks: {chunked_dataset.chunks}")

            if new_dataset.compression is None:
                print("new_dataset.compression is None")
            else:
                print("new_dataset.compression: {new_dataset.compression}")
        
            print(f"shape of {ds_name} after nan removal: {new_dataset.shape}")