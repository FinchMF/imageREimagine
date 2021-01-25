
"""
The trained network is the constant. Frozen weights act as the coefficents.

Loss is generated from weights in the target image // content and style 
- 
Content loss is generated from the content image and the target image

Stlye loss is generated from the style image the target image
-
The target image will start with a cloned weights of the content representation (although these could be randomized)
The target image's weights could also be set from a third image to find a content space between the content image and 'third' image


methods of generating alterations - 'third random' image "weights" -- adjusting weight influences on layers -- chosing different layers (change within or per model)

"""


import sys
import os
import datetime
import shutil
import model as m 
import utils



def neural_style(content_img, style_img, target_img=None, steps=5000, outfile=None, verbose=False):
    """
    Set device to GPU or CPU 
    """
    device = m.device
    # generate content tensor
    content = utils.load_image(content_img).to(device)
    # generate style tensor - reformat to match content shape
    style = utils.load_image(style_img, shape=content.shape[-2:]).to(device)
    # load pretrained VGG19 and set to device
    model = m.VGG19().network
    # generate content feature map
    content_features = utils.get_feature_maps(content, model)
    # generate style feature map
    style_features = utils.get_feature_maps(style, model)
    # generate style grams from style feature map
    style_grams = {layer: utils.gram_matrix(style_features[layer]) for layer in style_features}
    # if there is no target image, use the content image to set the weights of the target image
    if target_img == None:
        target = content.clone().requires_grad_(True).to(device)
    
    else:
        # otherwise load target image, generate tensor and map it to content shape
        target = utils.load_image(target_img, shape=content.shape[-2:]).to(device)
        target = target.requires_grad_(True).to(device)

    # train target image
    trained_image = utils.train_image(style_grams=style_grams,
                    content_features=content_features,
                    model=model,
                    Weights=m.Weights,
                    target=target,
                    steps=steps,
                    verbose=verbose)
    # set for naming output file
    if outfile == None:

        filename = 'trained_image.jpg'
    else:

        filename = outfile

    utils.save_image(trained_image=trained_image, filename=filename)

    if verbose:
        # generate gif of training process
        utils.make_training_gif()

        if not os.path.isdir('finished'):

            os.mkdir('finished')

        now = datetime.datetime.now()

        if not os.path.isdir(f'session_{now}'):

            os.mkdir(f'session_{now}')

        shutil.move('training', f'session_{now}/')
        shutil.move('training.gif', f'session_{now}/')
        shutil.move('trained_image.jpg', f'session_{now}')

        
        




    
        



if __name__ == '__main__':
    img_1, img_2 = sys.argv[1], sys.argv[2]
    neural_style(content_img=img_1, style_img=img_2) 


    







