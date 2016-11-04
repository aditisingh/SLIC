function [x1, y1] = correct_centre(x_c,y_c, labelled,img_ht, img_wd)
%gradient kernels
h1=[ 0 0 0; -1 0 1; 0 0 0];
h2=[ 0 -1 0 ; 0 0 0;0 1 0];

%gradient calculation
G1=conv2(labelled,h1);
G2=conv2(labelled,h2);


G1_ngbr=G1(max(y_c-1,1):min(y_c+1,img_ht),max(x_c-1,1):min(x_c+1,img_wd));
G2_ngbr=G2(max(y_c-1,1):min(y_c+1,img_ht),max(x_c-1,1):min(x_c+1,img_wd));
    
G=G1_ngbr.^2+G2_ngbr.^2;
    
[y1,x1]=find(G==min(min(G)));
    
if(size(x1,1)>1)
    x1=x_c; y1=y_c;
end

end

