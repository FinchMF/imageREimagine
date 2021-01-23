
import torch
from torchvision import models

# enable GPU or CPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG19:

    def __init__(self):
        # download pretrained model
        self.net = models.vgg19(pretrained=True).features
        # freeze network weights
        for param in self.net.parameters():
            param.requires_grad_(False)
        self.network = self.net.to(device)
        


class Weights:

    def __init__(self): 
        # style layers weighted
        # earlier layers are weighted to increase earlier representations influence
        self.style_weights = {

            'conv1_1': 1.,
            'conv2_1': 0.75,
            'conv3_1': 0.2,
            'conv4_1': 0.2,
            'conv5_1': 0.2,
        }
        # alpha
        self.content_weight = 1
        # beta
        self.style_weight = 1e6




        

        