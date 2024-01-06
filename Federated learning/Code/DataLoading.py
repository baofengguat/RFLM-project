import os
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
import scipy.sparse as sp
import torchvision.transforms as transforms

mean = [0.49139968, 0.48215841, 0.44653091]
stdv = [0.24703223, 0.24348513, 0.26158784]
transformTransfer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class DataSet_Loading(Dataset):
    def __init__(self, root ,args):

        self.root = root
        self.files = []

        # for class_label in os.listdir(os.path.join(self.root)):
        for patientClass in os.listdir(os.path.join(self.root)):
            if patientClass== args.category[0]:
                label=0
            elif patientClass== args.category[1]:
                label=1
            for patient in os.listdir(os.path.join(self.root,patientClass)):
                for name in os.listdir(os.path.join(self.root,patientClass,patient)):
                    img_file = os.path.join(self.root, "%s/%s/%s" % (patientClass,patient,name))
                    self.files.append({
                        "img": img_file,
                        "label": label,
                        "name": img_file
                    })


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imdecode(np.fromfile(datafiles["img"],dtype=np.uint8),cv2.IMREAD_COLOR)#
        # image.resize(224, 224)
        image = cv2.resize(image, (224,224))
        #image=image.transpose([2,0,1])

        label = datafiles["label"]
        #size = image.shape
        name = datafiles["name"]
        #image = np.asarray(image, np.float32)

        image = transformTransfer(image)

        return image, label, name

class CommonData_Loading(Dataset):
    def __init__(self, root ,args):
        self.root = root
        self.files = []
        # for class_label in os.listdir(os.path.join(self.root)):
        for patientClass in os.listdir(os.path.join(self.root)):
            for image in os.listdir(os.path.join(self.root,patientClass)):
                    img_file = os.path.join(self.root, "%s/%s" % (patientClass,image))
                    if patientClass == args.category[0]:
                        label = 0
                    elif patientClass == args.category[1]:
                        label = 1
                    self.files.append({
                        "img": img_file,
                        "label": label
                    })


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imdecode(np.fromfile(datafiles["img"],dtype=np.uint8),cv2.IMREAD_COLOR)#
        # image.resize(224, 224)
        # image = cv2.resize(image, (224,224))

        #image=image.transpose([2,0,1])
        # image = np.asarray(image, np.float32)

        label = datafiles["label"]

        image = transformTransfer(image)

        return image, label


def ServerLoading(args,flag=True):
    CenterName = os.listdir(os.path.join(args.original_dir))
    train_batches = []
    test_batches = []

    for index,SeverName in enumerate(CenterName):
        Data_classes = os.listdir(os.path.join(os.path.join(args.original_dir,SeverName)))
        Class_Path = os.path.join(os.path.join(args.original_dir,SeverName))
        for Data_class in Data_classes:
            if Data_class == 'train_data':
                train_dataset = DataSet_Loading(os.path.join(Class_Path,Data_class),args)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=flag,
                                          num_workers=args.num_workers)
            else:
                test_dataset = DataSet_Loading(os.path.join(Class_Path,Data_class),args)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if flag:
            print('Sever%d--%s Data has been completed' % (index, SeverName))
        assert train_loader
        assert test_loader

        train_batches.append(train_loader)
        test_batches.append(test_loader)

        A = np.random.rand(len(CenterName),len(CenterName))
        # A = np.eye(len(CenterName))

    return train_batches, test_batches,torch.tensor(normalize_adj(A), dtype=torch.float32)

def Common_Loading(args):
    CommonData = os.path.join(args.common_data)
    Common_dataset = CommonData_Loading(CommonData, args)
    Common_loader = DataLoader(Common_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)

    return Common_loader