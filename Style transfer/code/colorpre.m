g=imread('fox_input.jpg');
transfered=imread('fox_wave.jpg');
pre=colorpreservation(transfered,g);
imshow(pre);
imwrite(pre,'fox_colorpresrved.jpg');