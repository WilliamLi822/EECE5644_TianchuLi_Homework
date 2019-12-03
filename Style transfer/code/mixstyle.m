function [style1,style2,mix1,mix2] = mixstyle
f=imread('ancient_city.jpg');
ftransfer1=imread('city_transfer2.jpg');
ftransfer2=imread('city_transfer1.jpg');
mask1=imread('ancient_city_mask1.jpg');
mask2=imread('ancient_city_mask2.jpg');

f=imresize(mat2gray(f),1/4.8);
ftransfer1=mat2gray(ftransfer1);
ftransfer2=mat2gray(ftransfer2);
mask1=imresize(im2double(mask1),1/4.8);
mask2=imresize(im2double(mask2),1/4.8);

f1(:,:,1)=f(:,:,1).*mask1;
f1(:,:,2)=f(:,:,2).*mask1;
f1(:,:,3)=f(:,:,3).*mask1;
f3(:,:,1)=ftransfer1(:,:,1).*mask2;
f3(:,:,2)=ftransfer1(:,:,2).*mask2;
f3(:,:,3)=ftransfer1(:,:,3).*mask2;
style1=f1+f3;
figure,imshow(style1);
imwrite(style1,'city_style1.jpg');

f2(:,:,1)=f(:,:,1).*mask2;
f2(:,:,2)=f(:,:,2).*mask2;
f2(:,:,3)=f(:,:,3).*mask2;
f4(:,:,1)=ftransfer2(:,:,1).*mask1;
f4(:,:,2)=ftransfer2(:,:,2).*mask1;
f4(:,:,3)=ftransfer2(:,:,3).*mask1;
style2=f2+f4;
figure,imshow(style2);
imwrite(style2,'city_style2.jpg');

mix1=f3+f4;
figure,imshow(mix1);
imwrite(mix1,'city_mix1.jpg');

f5(:,:,1)=ftransfer1(:,:,1).*mask1;
f5(:,:,2)=ftransfer1(:,:,2).*mask1;
f5(:,:,3)=ftransfer1(:,:,3).*mask1;
f6(:,:,1)=ftransfer2(:,:,1).*mask2;
f6(:,:,2)=ftransfer2(:,:,2).*mask2;
f6(:,:,3)=ftransfer2(:,:,3).*mask2;
mix2=f5+f6;
figure,imshow(mix2);
imwrite(mix2,'city_mix2.jpg');
end

