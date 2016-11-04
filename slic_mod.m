clc; 
clear all; 
close all;

%READ MULTICHANNEL STACK
stack=load('stack.mat');

img1=stack.s.image;
[img_ht,img_wd]=size(img1);

img_ht=floor(img_ht/4);
img_wd=floor(img_wd/4);

N=img_ht*img_wd;    %number of pixels

prompt='How many superpixels?';
K=input(prompt);

labelled=zeros(img_ht,img_wd);

S=floor(sqrt(N/K));    %size of superpixel

%Setting parameters 
%Normalized weights

lambda_1 =0;%weight of P4110_CO_
lambda_2 =1/637.5191;% b=255, g=000, r=000; weight of P4110_C1_: GAD67
lambda_3 =1/1570.8;% b=255, g=000, r=000; weight of P4110_C2_: Parvalbumin
lambda_4 =1/1055.5;% b=123, g=255, r=000; weight of P4110_C4_: APC
lambda_5 =1/952.3659;% b=000, g=255, r=000; weight of P4110_C5_: RECA1
lambda_6 =1/3271.8;% b=000, g=255, r=255; weight of P4110_C6_: S100
lambda_7 =1/1804.6;%weight of P4110_C7_: NeuN
lambda_8 =1/785.4874;%weight of P4110_C8_: IBA1
lambda_9 =1/1114.8;% b=255, g=255, r=255; weight of P4110_C10: DAPI
lambda_10=0;%weight of P4111_C0_
lambda_11=1/3874.8;%weight of P4111_C1_: TubulinBeta3
lambda_12=1/1310.3;%weight of P4111_C2_: MAP2
lambda_13=1/7455.7;%weight of P4111_C5_: PLP
lambda_14=1/1843.4;%weight of P4111_C8_: GFAP
lambda_15=0;%weight of P4112_C0_ 
lambda_16=1/2907.1;%weight of P4112_C5_: PCNA
lambda_17=1/1268.2;%weight of P4112_C6_: CC3
lambda_18=1/5849.8;%weight of P4112_C7_: NFH
lambda_19=0;%weight of P4113_C0_
lambda_20=1/1853.4;%weight of P4113_C4_: Claretinin
lambda_21=1/11269;%weight of P4113_C5_: SynaptoPhys
lambda_22=1/10606;%weight of P4113_C6_: GLAST
lambda_23=1/17777;%weight of P4113_C7_: MBP
lambda_24=1/4387.4;%weight of P4113_C8_: TomatoLectin

m=10; %compactness control constant

count=1;
idx=randperm(ceil(img_ht/S)*ceil(img_wd/S));

%initial labelling
for i=1:S:img_wd
    for j=1:S:img_ht
        labelled(j:j+S-1,i:i+S-1)=idx(count);
        count=count+1;
    end
end

%fixing labelled matrix     
labelled=labelled(1:img_ht, 1:img_wd);

%get initial cluster centers
C_curr=zeros(size(unique(labelled),1),2);

vals=unique(labelled);
for i=1:size(vals,1)
    [x,y]=initial_centre(labelled,vals(i+1-min(min(labelled))));
    C_curr(i,1)=x; C_curr(i,2)=y;
    
    %correct centre in 3X3 neighbourhood
  [C_curr(i,1),C_curr(i,2)]=correct_centre(C_curr(i,1),C_curr(i,2),labelled,img_ht,img_wd);
        
end

num=30;
cnt=1;
threshold=0.05;
E=threshold;
err=[];

while cnt<=num%E>=threshold
    C_prev=C_curr;
    %distance calculation

    [D, dxy] = dist_calc(labelled,m, S, C_curr, stack, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, ...
    lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19, lambda_20, lambda_21, lambda_22, lambda_23, lambda_24);

    %assign pixels
    labelled= pixel_assignment(img_ht, img_wd, S, labelled, D, dxy, C_curr);
    
    C_curr=zeros(size(unique(labelled),1),2);

    %compute new centers
    vals=unique(labelled);
    for i=1:size(vals,1)
        [x,y]=initial_centre(labelled,vals(i+1-min(min(labelled))));
        C_curr(i,1)=x; C_curr(i,2)=y;
        %correct centre in 3X3 neighbourhood
        [C_curr(i,1),C_curr(i,2)]=correct_centre(C_curr(i,1),C_curr(i,2),labelled,img_ht, img_wd);
    end
    
    %error calculation
   	%E = residual_error_calc(C_curr, C_prev)
    %err =[err; E]
    cnt=cnt+1;
end

rgb=label2rgb(labelled);
imshow(rgb); 
imwrite(rgb,'labelled.png');
save('labelled.mat',rgb);
for lbl=1:nlabels
currenObject=(labelled==lbl);
labelled(imerode(currenObject,strel('disk',1)))=0;
end
rgb=label2rgb(labelled);
imshow(rgb); 