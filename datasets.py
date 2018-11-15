import torch
import torchvision.datasets as datasets
import torchvision.transforms as tf
import numpy as np


def prepare_mnist():
    # normalize data
    m = (0.1307,)
    st = (0.3081,)
    normalize = tf.Normalize(m, st)

    # load train data
    train_dataset = datasets.MNIST(
                        root='../data',
                        train=True,
                        transform=tf.Compose([tf.ToTensor(), normalize]),
                        download=True)

    # load test data
    test_dataset = datasets.MNIST(
                       root='../data',
                       train=False,
                       transform=tf.Compose([tf.ToTensor(), normalize]))

    return train_dataset, test_dataset

def sample_train(train_dataset, test_dataset, batch_size, k, n_classes, seed, shuffle_train=True, return_idxs=True):

    n = len(train_dataset)
    rrng = np.random.RandomState(seed)

    cpt = 0
    indices = torch.zeros(k)
    other = torch.zeros(n - k)
    card = k // n_classes

    for i in xrange(n_classes):
        class_items = (train_dataset.train_labels == i).nonzero().squeeze()
        n_class = len(class_items)
        rd = np.random.permutation(np.arange(n_class))
        indices[i * card: (i + 1) * card] = class_items[rd[:card]]
        other[cpt: cpt + n_class - card] = class_items[rd[card:]]
        cpt += n_class - card

    other = other.long()
    train_dataset.train_labels[other] = -1

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=16,
                                               shuffle=shuffle_train)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=16,
                                              shuffle=False)

    if return_idxs:
        return train_loader, test_loader, indices
    return train_loader, test_loader
