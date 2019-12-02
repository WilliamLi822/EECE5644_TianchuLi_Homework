function fpreserv = colorpreservation(f,style)
%implement color preservation
f=rgb2ntsc(f);
style=rgb2ntsc(style);
hist1=imhist(style(:,:,2));
hist2=imhist(style(:,:,3));
f(:,:,2)=histeq(f(:,:,2),hist1);
f(:,:,3)=histeq(f(:,:,3),hist2);
fpreserv=ntsc2rgb(f);
end

