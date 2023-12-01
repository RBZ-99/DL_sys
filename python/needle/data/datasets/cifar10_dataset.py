import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.imgs = []
        self.labels = []
        self.transforms = transforms

        for file_name in sorted(os.listdir(base_folder)):
            if train and "data_batch" in file_name:
                pass
            
            elif not train and "test_batch" in file_name:
                pass
            
            else:
                continue

            file_path = os.path.join(base_folder, file_name)
            data = unpickle(file_path)
            self.imgs.append(data[b'data'])
            self.labels += data[b'labels']

        self.imgs = np.concatenate(self.imgs) / 255.0
        self.labels = np.array(self.labels)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        flag = False

        if isinstance(index, int):
          flag = True
          index = [index]

        batch_imgs = self.imgs[index].reshape((len(index), 3, 32, 32))
        batch_labels = self.labels[index].reshape(-1)

        if self.transforms is not None:
            for transform in self.transforms:
               for ix, img in enumerate(batch_imgs):
                  batch_imgs[ix] = transform(batch_imgs[ix])

        if flag:
           return (batch_imgs[0], batch_labels[0])

        return (batch_imgs, batch_labels)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        # raise NotImplementedError()
        ### END YOUR SOLUTION
