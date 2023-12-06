from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, "rb") as f:
           magic, n, h, w = struct.unpack(">IIII", f.read(16))
           iter = struct.iter_unpack(">B", f.read())
           X = []

           for elem in iter:
              X.append(elem)
              
           X = np.resize(np.array(X), (n, h, w, 1)).astype(np.float32)
           X = X / 255.0
           
           with gzip.open(label_filename, "rb") as f:
              magic, n = struct.unpack(">II", f.read(8))
              iter = struct.iter_unpack(">B", f.read())
              y = []
              
              for elem in iter:
                 y.append(elem)
                 
              y = np.resize(np.array(y).astype(np.uint8), (n,))
              
        self.imgs = X
        self.labels = y
        self.transforms = transforms
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        flag = False

        if isinstance(index, int):
          flag = True
          index = [index]

        batch_imgs = self.imgs[index]
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
        ### BEGIN YOUR SOLUTION
        return len(self.imgs)
        # raise NotImplementedError()
        ### END YOUR SOLUTION