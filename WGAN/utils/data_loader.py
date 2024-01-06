import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from utils.CT_dataset import CT_dataset, CT_dataset_three_channels,MyDataset


def get_data_loader(args):
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if args.dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # train_dataset = MNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        # test_dataset = MNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # train_dataset = FashionMNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        # test_dataset = FashionMNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.CIFAR10(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.STL10(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'CTDATA':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_dataset = CT_dataset(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = CT_dataset(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == "CTDATA_three_channels":
        train_dataset = CT_dataset_three_channels(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = CT_dataset_three_channels(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == "lungsCT_three_channels":
        datarootTrain = r'data_txt\L3_train_1.txt'
        datarootTest = r'data_txt\L3_test_1.txt'
        train_dataset = MyDataset(names_file=datarootTrain, transform=False)
        test_dataset = MyDataset(names_file=datarootTest,  transform=False)
    #
    # # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader
