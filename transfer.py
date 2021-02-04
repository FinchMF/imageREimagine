
"""
The trained network is the constant. Frozen weights act as the coefficents.

Loss is generated from weights in the target image // content and style 
- 
Content loss is generated from the content image and the target image

Style loss is generated from the style image the target image (multiple style images could be used)
-
The target image will start with cloned weights of the content representation (although these could be randomized)
The target image's weights could also be set from a third image to find a content space between the content image and 'third' image


Methods of generating alterations - 'third random' image "weights" -- adjusting weight influences on layers -- chosing different layers (change within or per model)

"""

################
# DEPENDENCIES #
################
import sys
import os
import datetime
import shutil
import model as m 
import utils



def neural_style(content_img, alpha_style_img, beta_style_img=None, target_img=None, steps=5000, outfile=None, verbose=False):
    
    """
    generates an image using neural transfer technique 
    """

    #Set device to GPU or CPU 
    device = m.device
    # generate content tensor
    content = utils.load_image(content_img).to(device)
    # generate style tensor - reformat to match content shape
    alpha_style = utils.load_image(alpha_style_img, shape=content.shape[-2:]).to(device)
    # load pretrained VGG19 and set to device
    model = m.VGG19().network
    # generate content feature map
    content_features = utils.get_feature_maps(content, model)
    # generate style feature map
    alpha_style_features = utils.get_feature_maps(alpha_style, model)
    # generate style grams from style feature map
    alpha_style_grams = {layer: utils.gram_matrix(alpha_style_features[layer]) for layer in alpha_style_features}
    
    # if there is not a beta style image to train image on, beta style grams is None
    if beta_style_img == None:
        beta_style_grams = None
    # otherwise generate beta style grams
    else:
        beta_style = utils.load_image(beta_style_img, shape=content.shape[-2:]).to(device)
        beta_style_features = utils.get_feature_maps(beta_style, model)
        beta_style_grams = {layer: utils.gram_matrix(beta_style_features[layer]) for layer in beta_style_features}
    # if there is no target image, use the content image to set the weights of the target image
    if target_img == None:
        target = content.clone().requires_grad_(True).to(device)
    
    else:
        # otherwise load target image, generate tensor and map it to content shape
        target = utils.load_image(target_img, shape=content.shape[-2:]).to(device)
        target = target.requires_grad_(True).to(device)

    # train target image
    trained_image = utils.train_image(alpha_style_grams=alpha_style_grams,
                                      beta_style_grams=beta_style_grams,
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
            # check to see if directoy for finsihed sessions is built
            os.mkdir('finished')
        # set time for labeling session directory
        now = datetime.datetime.now()
    
        if not os.path.isdir(f'session_{now}'):
            # build session directory using timestamp
            os.mkdir(f'session_{now}')
        # move all generated materials to session folder
        shutil.move('training', f'session_{now}/')
        shutil.move('training.gif', f'session_{now}/')
        shutil.move(filename, f'session_{now}')
        shutil.move(f'session_{now}', 'finished')

        #STILL TO DO: move session folder into finished,
        # write logic to take trained image and style transfer it again 
        # figure out how to choose between different layers (content or style to see if there is an affect that can work)
        # generate combined logic with glitching library 



if __name__ == '__main__':
    # generic use case
    img_1, img_2 = sys.argv[1], sys.argv[2]
    neural_style(content_img=img_1, alpha_style_img=img_2) 


    







