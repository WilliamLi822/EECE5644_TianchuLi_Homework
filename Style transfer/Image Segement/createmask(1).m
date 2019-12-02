function [mask1,mask2] = createmask(f)
%create mask for given image
%using the dataset from matlab computer vision toolbox
data=load('triangleSegmentationNetwork');
net1=data.net;
[C,scores]=semanticseg(f,net1);
mask1=labeloverlay(f,C);
mask1=mat2gray(rgb2gray(mask1));
T=graythresh(mask1);
mask1(mask1<=T)=0;
mask1(mask1>T)=1;
figure,imshow(mask1);
imwrite(mask1,'ancient_city_mask1.jpg');
mask2=1-mask1;
figure,imshow(mask2);
imwrite(mask2,'ancient_city_mask2.jpg');
end

