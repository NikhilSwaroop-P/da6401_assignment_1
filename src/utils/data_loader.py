"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
from keras.datasets import mnist, fashion_mnist
import numpy as np

class Dataloader:
    def __init__(self,dataset_name, batch_size, shuffle=True):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        if dataset_name == "mnist":
            self.x_train, self.y_train, self.x_test, self.y_test = mnist.load_data()
        elif dataset_name == "fashion_mnist":
            self.x_train, self.y_train, self.x_test, self.y_test = fashion_mnist.load_data()
        else:
            raise ValueError("Invalid dataset name")

        # Reshape and normalize using vectorized operations
        self.x_train = self.x_train.astype("float32").reshape(self.x_train.shape[0], -1) / 255.0
        self.x_test = self.x_test.astype("float32").reshape(self.x_test.shape[0], -1) / 255.0
        self.y_train = self.y_train.astype("int")
        self.y_test = self.y_test.astype("int")
    def get_batch_train(self):
        num_samples = len(self.x_train)
        indices = np.arange(num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_x = self.x_train[batch_indices]
            batch_y = self.y_train[batch_indices]
            yield batch_x, batch_y

    def get_batch_test(self):
        num_samples = len(self.x_test)
        indices = np.arange(num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_x = self.x_test[batch_indices]
            batch_y = self.y_test[batch_indices]
            yield batch_x, batch_y