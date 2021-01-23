
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
import model as m 
import utils 

def neural_style(content_img, style_img, steps=5000):

    device = m.device

    content = utils.load_image(content_img).to(device)
    style = utils.load_image(style_img, shape=content.shape[-2:]).to(device)

    model = m.VGG19().network

    content_features = utils.get_feature_maps(content, model)
    style_features = utils.get_feature_maps(style, model)

    style_grams = {layer: utils.gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)


    trained_image = utils.train_image(style_grams=style_grams,
                    content_features=content_features,
                    model=model,
                    Weights=m.Weights,
                    target=target,
                    steps=steps)

    utils.save_image(trained_image=trained_image, filename='trained_image.jpg')



if __name__ == '__main__':
    img_1, img_2 = sys.argv[1], sys.argv[2]
    neural_style(content_img=img_1, style_img=img_2) 


    







