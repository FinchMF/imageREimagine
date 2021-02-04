# imageREimagine
Neural Transfer Style - Pytorch


**imageREimagine** 
- Recieves two images (unless you give it three):
    - one for content representation
    - one for style representation (unless you want to give it two)
- The content image is cloned to the target image and the style recreates the dimensions replicated with structural content.

- Returns an image, a gif of the training process and a folder of the training ouput  

## How to use It (quick)

*Transfer Style and Content between two images*

In Terminal: 

        $ python transfer.py <content image> <style image>

In Jupyter Notebook:
    
    import transfer

    transfer.neural_style(content_img=img_1, 
                          alpha_style_img=img_2,  
                          steps=10000, verbose=True)


*Transfer Style of two images and Content of a third image*

In Jupyter Notebook: 

    import transfer

    transfer.neural_style(content_img=img_1,
                          alpha_style_img=img_2,
                          beta_style_img=img_3,
                          steps=10_000_000,
                          verbose=True)


NOTE: step amount used in the examples are not set. Experiment!