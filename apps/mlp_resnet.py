import sys
sys.path.append('/Users/rushikeshzawar/Downloads/Personal/CMU_COURSES/dlsys/project/repo/DL_sys/python')

#sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    residual_fn = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    residual = nn.Residual(residual_fn)
    return nn.Sequential(residual, nn.ReLU())
    # raise NotImplementedError()
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    layers = [nn.Linear(dim, hidden_dim), nn.ReLU()]

    for i in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))

    layers.append(nn.Linear(hidden_dim, num_classes))

    return nn.Sequential(*layers)
    # raise NotImplementedError()
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()

    else:
        model.eval()
        
    loss_fn = nn.SoftmaxLoss()

    num_batches = 0
    avg_err = 0.0
    avg_loss = 0.0

    for i, batch in enumerate(dataloader):
        batch_x, batch_y = batch[0], batch[1]
        
        # batch_x.device = model.device
        # batch_y.device = model.device

        if opt is not None:
            opt.reset_grad()

        logits = model(nn.Flatten()(batch_x))
        loss = loss_fn(logits, batch_y)

        if opt is not None:
            loss.backward()
            opt.step()

        avg_err += len(np.where(np.argmax(logits.numpy(), (1)) != batch_y.numpy())[0])
        avg_loss += loss.numpy()
        num_batches += 1

    return avg_err / len(dataloader.dataset), avg_loss / num_batches
    # raise NotImplementedError()
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_imgs_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    mnist_train_dataset = ndl.data.MNISTDataset(train_imgs_path, train_labels_path)
    train_loader = ndl.data.DataLoader(dataset = mnist_train_dataset, batch_size = batch_size, shuffle = True)

    test_imgs_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    mnist_test_dataset = ndl.data.MNISTDataset(test_imgs_path, test_labels_path)
    test_loader = ndl.data.DataLoader(dataset = mnist_test_dataset, batch_size = batch_size, shuffle = False)

    model = nn.Fin_FFC() #MLPResNet(784, hidden_dim = hidden_dim)
    opt = optimizer(model.parameters(), lr = lr, weight_decay = weight_decay)

    train_err, train_loss = 0.0, 0.0
    for curr_epoch in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)

    test_err, test_loss = epoch(test_loader, model)

    return (train_err, train_loss, test_err, test_loss)
    # raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
