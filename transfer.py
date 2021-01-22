
"""
The trained network is the constant. Frozen weights act as the coefficents.

Loss is generated from weights in the target image // content and style 
- 
Content loss is generated from the content image and the target image

Stlye loss is generated from the style image the target image
-
The target image will start with a cloned weights of the content representation (although these could be randomized)
The target image's weights could also be set from a third image to find a content space between the content image and 'third' image

"""


