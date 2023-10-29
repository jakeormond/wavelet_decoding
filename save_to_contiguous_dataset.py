import h5py

input_data_path = "/media/jake/LaCie/wave_data_05_05_2023/wave_data_new_concat.h5"
output_data_path = "/media/jake/LaCie/wave_data_05_05_2023/wave_data_contiguous.h5"

# datasets to copy
# ds_names = ['wave_mad_norm', 'position', 'head_direction']
ds_names = ['position', 'hd']

# Open the existing HDF5 file in read mode
with h5py.File(input_data_path, 'r') as f:
    

    for ds_name in ds_names:
        # Open the existing chunked dataset
        chunked_dataset = f[ds_name]

        # Get the shape and datatype of the existing dataset
        shape = chunked_dataset.shape
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
                new_dataset = new_f.create_dataset(ds_name, shape, dtype=dtype)

            if ds_name == 'wave_mad_norm':
                          # Copy data from the chunked dataset to the new contiguous dataset
                for i in range(shape[0]):
                    # Read a chunk from the chunked dataset
                    chunk = chunked_dataset[i, :, :]

                    # Write the chunk to the new dataset at the corresponding location
                    new_dataset[i, :, :] = chunk
                    print(f"saved chunk {i}")

            else:
                new_dataset[...] = chunked_dataset

            if new_dataset.chunks is None:
                print("new_dataset.chunks is None")
            else:
                print("new_dataset.chunks: {chunked_dataset.chunks}")

            if new_dataset.compression is None:
                print("new_dataset.compression is None")
            else:
                print("new_dataset.compression: {new_dataset.compression}")
        