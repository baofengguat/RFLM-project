from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import numpy as np
import cv2
import torchvision.transforms as transforms

class CT_dataset(data.Dataset):    
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training_full.pt'
    test_file = 'test_full.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.prepare_from_file()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' Please check again.')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
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

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def prepare_from_file(self):
        if self._check_exists():
            return
            
        # prepare "raw" and "processed" directories
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        
        # get image and label from file on disk
        print("Processing...")

        training_set = self.parse_data_from_file(os.path.join(self.root, self.raw_folder, "dataset.npy"), is_train=True)
        test_set = self.parse_data_from_file(os.path.join(self.root, self.raw_folder, "dataset.npy"), is_train=False)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print("Done!")

    def parse_data_from_file(self, filepath, is_train):
        raw_data = np.load(filepath, allow_pickle=True).item()
        images, labels = raw_data["X"], raw_data["y"]

        if is_train:
            real_images = images[:6100]
            real_labels = labels[:6100]
        else:
            real_images = images[6100:]
            real_labels = labels[6100:]

        real_images = torch.ByteTensor(real_images).view(-1, 224, 224)
        real_labels = torch.LongTensor(real_labels)
        
        return (real_images, real_labels)


class CT_dataset_three_channels(data.Dataset):    
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.prepare_from_file()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' Please check again.')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.numpy().reshape(3, 224, 224)
        img = np.stack([img[0, :, :], img[1, :, :], img[2, :, :]], axis=2)
        img = Image.fromarray(img, 'RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def prepare_from_file(self):
        if self._check_exists():
            return
            
        # prepare "raw" and "processed" directories
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        
        # get image and label from file on disk
        print("Processing...")

        training_set = self.create_dataset_from_three_channel_png(os.path.join(self.root, self.raw_folder, "train_data/LTB"), is_train=True)
        test_set = self.create_dataset_from_three_channel_png(os.path.join(self.root, self.raw_folder, "train_data/LTB"), is_train=False)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print("Done!")

    def create_dataset_from_three_channel_png(self, file_dir, is_train):
        real_images = []
        real_labels = []

        # currently only deal with benign samples
        for (root, patient_names, _) in os.walk(file_dir):
            for current_patient in patient_names:
                img_root = os.path.join(root, current_patient)
                for (_, _, img_names) in os.walk(img_root):
                    for current_img in img_names:
                        img = cv2.imread(os.path.join(img_root, current_img))
                        real_images.append(img)
                        real_labels.append(0)

        if is_train:
            real_images = real_images[:500]
            real_labels = real_labels[:500]
        else:
            real_images = real_images[500:]
            real_labels = real_labels[500:]

        real_images = torch.ByteTensor(real_images).view(-1, 3, 224, 224)
        real_labels = torch.LongTensor(real_labels)
        
        return (real_images, real_labels)

# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])
transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
])


class MyDataset(data.Dataset):

    def __init__(self, names_file, transform=None):

        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.names_list[idx].split('*')[0]
        if not os.path.isfile(image_path):
            print(image_path + ' ' + 'does not exist!')
            return None

        image = Image.open(image_path)
        image = image.convert("RGB")

        if image.mode == 'L':
            image = transform(image)
        else:
            image = transform1(image)

        label = int(self.names_list[idx].split('*')[2])

        return (image, label)


if __name__ == "__main__":
    custom_data = CT_dataset(os.path.join(os.getcwd(), "datasets/CT-data"))