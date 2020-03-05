import numpy as np
import tensorflow as tf
import torch
import torch.utils.data as data
import copy    
    
def noisy_label_dataset(val_ratio,noise_ratio):
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_val = np.moveaxis(x_train, [3, 1, 2], [1, 2, 3]).astype('float32')[len(x_train)-int(len(x_train)*val_ratio):len(x_train)]
    x_train = np.moveaxis(x_train, [3, 1, 2], [1, 2, 3]).astype('float32')[:len(x_train)-int(len(x_train)*val_ratio)]
    x_test = np.moveaxis(x_test, [3, 1, 2], [1, 2, 3]).astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    label_list = [0,1,2,3,4,5,6,7,8,9]
    
    
    y_val = y_train.reshape(-1).astype('long')[len(x_train):]
    y_val_rand = copy.copy(y_val)
    for i in range(int(len(y_val)*noise_ratio)):
        removedlist = copy.copy(label_list)
        removedlist.remove(y_val_rand[i])
        y_val_rand[i] = np.random.choice(removedlist)

    
    y_train = y_train.reshape(-1).astype('long')[:len(x_train)]
    y_train_rand = copy.copy(y_train)
    for i in range(int(len(y_train)*noise_ratio)):
        removedlist = copy.copy(label_list)
        removedlist.remove(y_train_rand[i])
        y_train_rand[i] = np.random.choice(removedlist)

    y_test = y_test.reshape(-1).astype('long')


    ds_prob = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_rand))
    dataloader_prob = data.DataLoader(dataset=ds_prob, batch_size=1000, shuffle=False)

    ds_val = data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val_rand))
    dataloader_val = data.DataLoader(dataset=ds_val, batch_size=1000, shuffle=False)

    ds_test = data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    dataloader_test = data.DataLoader(dataset=ds_test, batch_size=1000, shuffle=True)
    
    return dataloader_prob, dataloader_val, dataloader_test, x_train,y_train_rand


