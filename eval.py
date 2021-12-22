import numpy as np

## check whether two arrays equals

if __name__ == "__main__":
    arr1_path = '/home/jodie/torchsde/torch_data.npy'
    arr2_path = '/home/jodie/tfsde/tf_data.npy'

    arr1 = np.load(arr1_path, 'r')
    arr2 = np.load(arr2_path, 'r')


    print('\n')
    print('path1: ', arr1_path)
    print('path2: ', arr2_path)
    print(np.sum(np.abs(arr1-arr2)))
    print('\n')
