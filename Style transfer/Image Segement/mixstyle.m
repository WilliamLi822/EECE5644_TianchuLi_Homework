function [style1,style2,mix] = mixstyle(f,ftransfer1,ftransfer2)
f=mat2gray(f);
ftransfer1=mat2gray(ftransfer1);
ftransfer2=mat2gray(ftransfer2);

f1(:,:,1)=f(:,:,1).*mask1;
f1(:,:,2)=f(:,:,2).*mask1;
f1(:,:,3)=f(:,:,3).*mask1;
f3(:,:,1)=ftransfer1(:,:,1).*mask2;
f3(:,:,2)=ftransfer1(:,:,2).*mask2;
f3(:,:,3)=ftransfer1(:,:,3).*mask2;
style1=f1+f3;
figure,imshow(style1);

f2(:,:,1)=f(:,:,1).*mask1;
f2(:,:,2)=f(:,:,2).*mask1;
f2(:,:,3)=f(:,:,3).*mask1;
f4(:,:,1)=ftransfer2(:,:,1).*mask1;
f4(:,:,2)=ftransfer2(:,:,2).*mask1;
f4(:,:,3)=ftransfer2(:,:,3).*mask1;
style2=f2+f4;
figure,imshow(style2);

mix=f3+f4;
figure,imshow(mix);
end

