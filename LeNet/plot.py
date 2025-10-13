from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms

train_data = FashionMNIST(root='./data', 
                          train=True, 
                          download=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))   
