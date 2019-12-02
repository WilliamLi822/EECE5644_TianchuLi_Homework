f=imread('D:\file\Introduction to machine learning\term project\cat_sketch.jpg');
style=imread('D:\file\Introduction to machine learning\term project\style_sketch.jpg');
preserved=colorpreservation(f,style);
figure,imshow(preserved);
imwrite(preserved,'colorpreserved1.jpg');
f1=rgb2gray(f);
figure,imshow(f1);
imwrite(f1,'colorpreserved2.jpg');