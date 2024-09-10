#import statements
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda #ToTensor = transform that converts images to tensors. Lambda = allows you to create custom transforms

#Loading dataset 
ds = datasets.FashionMNIST(
    root="data",
    train=True, #Training set
    download=True, 
    transform=ToTensor(), #Transforms to a tensor
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) #Creates tensor of zeros with size 10. (One for each class in fashion mnst), updates this tensor to a one-hot encoded vector. For the given label y, it sets the corresponding index to 1.
    #Note to self, study Lambda transforms more.
)