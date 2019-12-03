f=imread('cat_sketch.jpg');
style=imread('style_sketch.jpg');
preserved=colorpreservation(f,style);
figure,imshow(preserved);
imwrite(preserved,'fox_sketch1.jpg');
f1=rgb2gray(f);
figure,imshow(f1);
imwrite(f1,'fox_sketch2.jpg');