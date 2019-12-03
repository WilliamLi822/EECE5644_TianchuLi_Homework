To implement all the MATLAB code, computer vision toolbox is needed.
createmask.m is used to create mask for a given image. The segmentation network we used is triangleSegmentationNetwork.mat
mixstyle.m is used to mix different style using the created mask
colorpreservation.m is the function to preserve the color from the original image
sketch.m is to promise the transferred image looks like a black&white image

Imagenet-vgg-verydeep-19.mat is the pre-trained model used to generate the result. It is available at https://www.kaggle.com/teksab/imagenetvggverydeep19mat. The file should be put in the same fold with .py file