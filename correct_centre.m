function [x1, y1] = correct_centre(x_c,y_c, labelled)
%gradient kernels
h1=[ 0 0 0; -1 0 1; 0 0 0];
h2=[ 0 -1 0 ; 0 0 0;0 1 0];

%gradient calculation
G1=conv2(labelled,h1);
G2=conv2(labelled,h2);

G1_ngbr=G1(y_c-1:y_c+1,x_c-1:x_c+1);
G2_ngbr=G2(y_c-1:y_c+1,x_c-1:x_c+1);
    
G=G1_ngbr.^2+G2_ngbr.^2;
    
[y1,x1]=find(G==min(min(G)));
    
if(size(x1,1)>1)
    x1=x_c; y1=y_c;
end

end

