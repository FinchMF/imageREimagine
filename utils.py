
from PIL import Image
from io import BytesIO

import imageio
import glob

import os
import requests
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torchvision import transforms



def load_image(img_path, max_size=400, shape=None):

    """
    Load transform an image to tensor. 
    Make sure the image is <= 400 pixels in x-y dims
    """

    if "http" in img_path:
        # if the image is a url
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')

    else:

        image = Image.open(img_path).convert('RGB')
    
    # adjust the size if needed
    if max(image.size) > max_size:

        size = max_size

    else:

        size = max(image.size)

    if shape is not None:

        size = shape
    # transform image into tensor
    transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                          (0.229, 0.224, 0.225))
    ])
    # remove transparent alpha channel and add batch dimension
    image = transform(image)[:3,:,:].unsqueeze(0)

    return image



def img_convert(tensor):

    """
    transform tensor to an image
    """
    # set tensor to cpu
    image = tensor.to('cpu').clone().detach()
    # set tensor to numpy
    image = image.numpy().squeeze()
    # transform and normalize pixels
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0,1)

    return image



def get_feature_maps(image, model, layers=None):

    """
    Image -> Foward Pass through Model = feature maps for set of layers

    Defaults match VGG19 Gatys et al (2016)
    """

    if layers is None:
        # layers needed for style and content representations
        layers = {

            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2', # layer to be used to capture content representation
            '28': 'conv5_1'
        }

    features = {}
    x = image
    # pytorch pretrained models have dict holding each module in the module
    for name, layer in model._modules.items():
        # passing image through each module layer
        x = layer(x)
        if name in layers:
            # set features maps
            features[layers[name]] = x

    return features


def gram_matrix(tensor):

    """
    Calculate gram matrix of a feature map to extract style representations
    """

    # ignore batch, get depth, height and width 
    _, d, h, w = tensor.size()
    # reshape in order to focus on multiplication of features for each channel
    tensor = tensor.view(d, h * w)
    # calculate gram matrx
    gram = torch.mm(tensor, tensor.t())

    return gram

    

def train_image(style_grams, content_features, model, Weights, target, steps=5000, show_every=400, verbose=False):

    """
    Trains target image
    """
    # set optimizer
    optimizer = optim.Adam([target], lr=0.003)
    x = 0
    for ii in range(1, steps+1):
        # extract features maps
        target_features = get_feature_maps(target, model)
        # calculate content loss on layer conv4_2 of VGG19 - need to change if another model is used
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0
        # calculate style loss
        for layer in Weights().style_weights:
            # generate feature map
            target_feature = target_features[layer]
            # generate gram matrix from layer's feature map
            target_gram = gram_matrix(target_feature)
            # remove batch (translucent dimension)
            _, d, h, w = target_feature.shape
            #  call style gram matrix from 'teacher image'
            style_gram = style_grams[layer]
            # magnify average of error per layer to generate style loss of layer
            layer_style_loss = Weights().style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            # add to overall style loss
            style_loss += layer_style_loss / (d * h * w)
        # compute total loss
        total_loss = Weights().content_weight * content_loss + Weights().style_weight * style_loss
        # backpropogate and optimize total loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ii % show_every == 0:
            # show progress
            print(f'Total Loss:  {total_loss.item()}')
            plt.imshow(img_convert(target))
            plt.show()
            
            if verbose:

                if not os.path.isdir('training'):
                    
                    os.mkdir('training')

                plt.imsave(img_convert(target), f'training/training_img_{x}.jpg')
                x += 1

    return img_convert(target)


def save_image(trained_image, filename):
    """
    saves image using matplotlib

    need to transition to PIL
    """
    plt.imsave(filename, trained_image)

    print(f'Image saved at {filename}')


def make_training_gif():

    filenames = glob.glob('training/*.jpg')
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('training.gif', images)

    print('Training GIF made')

    



