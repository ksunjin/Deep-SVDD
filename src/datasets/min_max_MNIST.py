import torch
import numpy as np
from torch.utils.data import Subset
import torchvision.transforms as transforms


def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x

from torchvision.datasets import ImageFolder, MNIST


def min_max_cal():
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    #train_set_full = MyMNIST(root='../../data/MyMNIST', transform=train_transforms)
    train_set_full = MyMNIST(root='../../data/MyMNIST', train=True, download=True,
                             transform=None, target_transform=None)

    MIN = []
    MAX = []
    for normal_classes in range(10):

        train_idx_normal = get_target_label_idx(train_set_full.train_labels.clone().data.cpu().numpy(), normal_classes)
        print(train_idx_normal)
        train_set = Subset(train_set_full, train_idx_normal)

        _min_ = []
        _max_ = []

        for idx in train_set.indices:
            gcm = global_contrast_normalization(train_set.dataset.data[idx].float(), 'l1')
            _min_.append(gcm.min())
            _max_.append(gcm.max())
        MIN.append(np.min(_min_))
        MAX.append(np.max(_max_))
    print(list(zip(MIN, MAX)))


class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
if __name__ == '__main__':
    min_max_cal()