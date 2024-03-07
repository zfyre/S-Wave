import torch
import os
import torchvision
import torchvision.transforms as transforms

def get_dataset(dataset_name):

    rootdir = './data'
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)

    # Step 1: Load the custom dataset
    if dataset_name.lower() == "stl10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.STL10(root=rootdir, split='train',
                                               download=True, transform=transform)
        testset = torchvision.datasets.STL10(root=rootdir, split='test',
                                              download=True, transform=transform)
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root=rootdir, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=rootdir, train=False,
                                        download=True, transform=transform)

    elif dataset_name.lower() == "imagenet":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.ImageNet(root=rootdir, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.ImageNet(root=rootdir, train=False,
                                        download=True, transform=transform)

    # Step 2: Extract image data from the datasets
    train_data = torch.stack([img for img, _ in trainset], dim=0)
    test_data = torch.stack([img for img, _ in testset], dim=0)

    # Step 3: Save training and testing data into separate .pth files
    torch.save(train_data, f'{rootdir}/{dataset_name.lower()}_train.pth')
    torch.save(test_data, f'{rootdir}/{dataset_name.lower()}_test.pth')

    print(f"{dataset_name} data tensors saved successfully!")


if __name__ == "__main__":
    get_dataset('stl10')
    # get_dataset('cifar10')