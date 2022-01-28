from torch.utils.data import Subset
from PIL import Image
# from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np
import torchvision.transforms as transforms
import os
import torchvision
from torchvision.datasets import ImageFolder

class Casting_Dataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class=0):
        super().__init__(root)
        assert normal_class in [0, 1]
        self.root = root
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = normal_class
        self.outlier_classes = 1 - normal_class

        #transform.Resize (X)
        min_max = [(-3.6562, 3.3021572), (-3.6562, 3.3513894)]

        #transform.Resize(size = (128,128))
        #min_max = [(-3.674326, 3.0845926), (-3.675603, 3.129845)]
        
        train_transforms = transforms.Compose([
            transforms.Resize(size=(128,128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((.5642, 0.5642, 0.5642), (0.2369, 0.2369, 0.2369))
            transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_max[normal_class][0]],
                                 [min_max[normal_class][1] - min_max[normal_class][0]])
        ])

        raw_transforms = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(size=(128,128)),
            transforms.ToTensor(),
            transforms.Normalize((.5642, 0.5642, 0.5642), (0.2369, 0.2369, 0.2369))
        ])

        train_dataset = MyImage(root=os.path.join(self.root, 'train'), transform=train_transforms)
        test_dataset = MyImage(root=os.path.join(self.root, 'test'), transform=test_transforms, train=False,
                               raw_transform=raw_transforms)

        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_dataset.targets, self.normal_classes)
        self.train_set = Subset(train_dataset, train_idx_normal)
        self.test_set = test_dataset

class MyImage(ImageFolder):
    def __init__(self, train=True, raw_transform=None, *args, **kwargs):
        super(MyImage,self).__init__(*args, **kwargs)
        self.train = train
        self.raw_transform = raw_transform
        self.make_dataarray()

    def make_dataarray(self):
        arrl = []
        for fpth, _ in self.imgs:
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