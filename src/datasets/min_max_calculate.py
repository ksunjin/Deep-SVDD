import torch
import numpy as np
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

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

def min_max_print():
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(128,128))
    ])

    train_set_full_c = MyImage(root='../../data/marble_data', transform=train_transforms)

    MIN = []
    MAX = []
    for normal_classes in range(10):

        train_idx_normal = get_target_label_idx(train_set_full_c.targets, normal_classes)
        train_set = Subset(train_set_full_c, train_idx_normal)

        _min_ = []
        _max_ = []

        for idx in train_set.indices:
            ndarr = train_set.dataset.train_data[idx]
            tensor = torch.tensor(ndarr)
            gcm = global_contrast_normalization(tensor, 'l1')
            _min_.append(gcm.min())
            _max_.append(gcm.max())
        try:
            MIN.append(np.min(_min_))
            MAX.append(np.max(_max_))
        except ValueError:
            pass
    print(list(zip(MIN, MAX)))

class MyImage(ImageFolder):
    def __init__(self, train=True, raw_transform=None, *args, **kwargs):
        super(MyImage, self).__init__(*args, **kwargs)
        self.train = train
        self.raw_transform = raw_transform
        self.make_dataarray()

    def make_dataarray(self):
        arrl = []
        for fpth, _ in self.imgs:
            # print(self.loader(fpth))
            if self.raw_transform is not None:
                arrl.append(self.raw_transform(self.loader(fpth)).detach().cpu().numpy()[np.newaxis])
            else:
                if self.transform is not None:
                    arrl.append(self.transform(self.loader(fpth)).detach().cpu().numpy()[np.newaxis])
        if self.train:
            self.train_data = np.concatenate(arrl)
        else:
            self.test_data = np.concatenate(arrl)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


if __name__ == '__main__':
    min_max_print()