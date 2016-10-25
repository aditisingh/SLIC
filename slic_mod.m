clc; 
clear all; 
close all;

%READ MULTIDIMENSIONAL STACK
stack=load('stack.mat');

img1=stack.s.image;
[img_ht,img_wd]=size(img1);

N=img_ht*img_wd;    %number of pixels

prompt='How many superpixels?';
K=input(prompt);

labelled=zeros(img_ht,img_wd);

S=floor(sqrt(N/K));    %size of superpixel

%Setting parameters
lambda_1=1;%weight of P4110_CO
lambda_2=1;%weight of P4110_C1: GAD67
lambda_3=1;%weight of GAD67
lambda_4=1;%weight of GAD67
lambda_5=1;%weight of GAD67
lambda_6=1;%weight of GAD67
lambda_7=1;%weight of GAD67
lambda_8=1;%weight of GAD67
lambda_9=1;%weight of GAD67
lambda_10=1;%weight of GAD67
lambda_11=1;%weight of GAD67
lambda_12=1;%weight of GAD67
lambda_13=1;%weight of GAD67
lambda_14=1;
lambda_15=1;
lambda_16=1;
lambda_17=1;
lambda_18=1;
lambda_19=1;
lambda_20=1;
lambda_21=1;
lambda_22=1;
lambda_23=1;
lambda_24=1;

count=1;

%initial labelling
for i=1:S:img_wd
    for j=1:S:img_ht
        for x=0:S-1
            for y=0:S-1
                labelled(j+y,i+x)=count;
            end
        end
        count=count+1;
    end
end

%get initial cluster centers
C=zeros(size(unique(labelled),1),2);

%gradient kernels
h1=[ 0 0 0; -1 0 1; 0 0 0];
h2=[ 0 -1 0 ; 0 0 0;0 1 0];

%gradient calculation
G1=conv2(labelled,h1);
G2=conv2(labelled,h2);

for i=1:size(unique(labelled),1)
    [x,y]=initial_centre(labelled,i);
    C(i,1)=x; C(i,2)=y;
    
    G1_ngbr=G1(C(i,1)-1:C(i,1)+1,C(i,2)-1:C(i,2)+1);
    G2_ngbr=G2(C(i,1)-1:C(i,1)+1,C(i,2)-1:C(i,2)+1);
    
    G=G1_ngbr.^2+G2_ngbr.^2;
    
    [x1,y1]=find(G==min(min(G)));
    if(size(x1,1)==1)
        C(i,1)=x1; C(i,2)=y1;
    end
        
end



