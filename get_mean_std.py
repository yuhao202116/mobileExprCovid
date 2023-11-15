import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assuming images are 3 x H x W (3 channels)
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.Grayscale(),
    transforms.ToTensor()])
dataset = datasets.ImageFolder(root='D:\\deep_learning\\Mobilenet\\data_set\\lung_data\\lung_photos',
                               transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

mean = 0.
std = 0.
nb_samples = 0.

pbar = tqdm(dataloader)
for data, _ in pbar:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

    pbar.set_description('Progress')

mean /= nb_samples
std /= nb_samples

print(f'Mean: {mean}')
print(f'Std: {std}')
