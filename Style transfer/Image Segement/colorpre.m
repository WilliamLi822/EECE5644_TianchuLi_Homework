g=imread('D:\file\Introduction to machine learning\term project\fox_wave.jpg');
transfered=imread('D:\file\Introduction to machine learning\term project\fox_wave.jpg');
pre=colorpreservation(transfered,g);
imshow(pre);
imwrite(pre,'fox_colorpresrved.jpg');