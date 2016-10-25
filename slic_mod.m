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

lambda_1 =0;%weight of P4110_CO_
lambda_2 =1;%weight of P4110_C1_: GAD67
lambda_3 =1;%weight of P4110_C2_: Parvalbumin
lambda_4 =1;%weight of P4110_C4_: APC
lambda_5 =1;%weight of P4110_C5_: RECA1
lambda_6 =1;%weight of P4110_C6_: S100
lambda_7 =1;%weight of P4110_C7_: NeuN
lambda_8 =1;%weight of P4110_C8_: IBA1
lambda_9 =1;%weight of P4110_C10: DAPI
lambda_10=0;%weight of P4111_C0_
lambda_11=1;%weight of P4111_C1_: TubulinBeta3
lambda_12=1;%weight of P4111_C2_: MAP2
lambda_13=1;%weight of P4111_C5_: PLP
lambda_14=1;%weight of P4111_C8_: GFAP
lambda_15=0;%weight of P4112_C0_ 
lambda_16=1;%weight of P4112_C5_: PCNA
lambda_17=1;%weight of P4112_C6_: CC3
lambda_18=1;%weight of P4112_C7_: NFH
lambda_19=0;%weight of P4113_C0_
lambda_20=1;%weight of P4113_C4_: Claretinin
lambda_21=1;%weight of P4113_C5_: SynaptoPhys
lambda_22=1;%weight of P4113_C6_: GLAST
lambda_23=1;%weight of P4113_C7_: MBP
lambda_24=1;%weight of P4113_C8_: TomatoLectin

m=10; %compactness control constant

count=1;

img1=stack.s(1).image;
img2=stack.s(2).image;
img3=stack.s(3).image;
img4=stack.s(4).image;
img5=stack.s(5).image;
img6=stack.s(6).image;
img7=stack.s(7).image;
img8=stack.s(8).image;
img9=stack.s(9).image;
img10=stack.s(10).image;
img11=stack.s(11).image;
img12=stack.s(12).image;
img13=stack.s(13).image;
img14=stack.s(14).image;
img15=stack.s(15).image;
img16=stack.s(16).image;
img17=stack.s(17).image;
img18=stack.s(18).image;
img19=stack.s(19).image;
img20=stack.s(20).image;
img21=stack.s(21).image;
img22=stack.s(22).image;
img23=stack.s(23).image;
img24=stack.s(24).image;

%initial labelling
for i=1:S:img_wd
    for j=1:S:img_ht
        labelled(j:j+S-1,i:i+S-1)=count;
        count=count+1;
    end
end

%fixing labelled matrix 
labelled=labelled(1:img_ht, 1:img_wd);

%get initial cluster centers
C=zeros(size(unique(labelled),1),2);


for i=1:size(unique(labelled),1)
    [x,y]=initial_centre(labelled,i+1-min(min(labelled)));
    C(i,1)=x; C(i,2)=y;
    
    %correct centre in 3X3 neighbourhood
  [C(i,1),C(i,2)]=correct_centre(C(i,1),C(i,2),labelled);
        
end

%distance calculation
D=zeros(img_ht, img_wd, size(unique(labelled),1)); %distance for each pixel from each centre

for i=1:size(unique(labelled),1)
    for j=1:img_ht
        for k=1:img_wd
            d1=(img1(j,k)-img1(C(i,2),C(i,1))).^2;
            d2=(img2(j,k)-img2(C(i,2),C(i,1))).^2;
            d3=(img3(j,k)-img3(C(i,2),C(i,1))).^2;
            d4=(img4(j,k)-img4(C(i,2),C(i,1))).^2;
            d5=(img5(j,k)-img5(C(i,2),C(i,1))).^2;
            d6=(img6(j,k)-img6(C(i,2),C(i,1))).^2;
            d7=(img7(j,k)-img7(C(i,2),C(i,1))).^2;
            d8=(img8(j,k)-img8(C(i,2),C(i,1))).^2;
            d9=(img9(j,k)-img9(C(i,2),C(i,1))).^2;
            d10=(img10(j,k)-img10(C(i,2),C(i,1))).^2;
            d11=(img11(j,k)-img11(C(i,2),C(i,1))).^2;
            d12=(img12(j,k)-img12(C(i,2),C(i,1))).^2;
            d13=(img13(j,k)-img13(C(i,2),C(i,1))).^2;
            d14=(img14(j,k)-img14(C(i,2),C(i,1))).^2;
            d15=(img15(j,k)-img15(C(i,2),C(i,1))).^2;
            d16=(img16(j,k)-img16(C(i,2),C(i,1))).^2;
            d17=(img17(j,k)-img17(C(i,2),C(i,1))).^2;
            d18=(img18(j,k)-img18(C(i,2),C(i,1))).^2;
            d19=(img19(j,k)-img19(C(i,2),C(i,1))).^2;
            d20=(img20(j,k)-img20(C(i,2),C(i,1))).^2;
            d21=(img21(j,k)-img21(C(i,2),C(i,1))).^2;
            d22=(img22(j,k)-img22(C(i,2),C(i,1))).^2;
            d23=(img23(j,k)-img23(C(i,2),C(i,1))).^2;
            d24=(img24(j,k)-img24(C(i,2),C(i,1))).^2;  
            
            dxy=((k-C(i,1)).^2+(j-C(i,2)).^2);
            
            D(j,k,i)=sqrt(double(d1)*(lambda_1).^2 + double(d2)*(lambda_2).^2 + double(d3)*(lambda_3).^2 + double(d4)*(lambda_4).^2 + double(d5)*(lambda_5).^2 ... 
                + double(d6)*(lambda_6).^2 + double(d7)*(lambda_7).^2 + double(d8)*(lambda_8).^2 + double(d9)*(lambda_9).^2 + double(d10)*(lambda_10).^2 ...
                + double(d11)*(lambda_11).^2 + double(d12)*(lambda_12).^2 + double(d13)*(lambda_13).^2 + double(d14)*(lambda_14).^2 + double(d15)*(lambda_15).^2 ...
                + double(d16)*(lambda_16).^2 + double(d17)*(lambda_17).^2 + double(d18)*(lambda_18).^2 + double(d19)*(lambda_19).^2 + double(d20)*(lambda_20).^2 ...
                + double(d21)*(lambda_21).^2 + double(d22)*(lambda_22).^2 + double(d23)*(lambda_23).^2 + double(d24)*(lambda_24).^2 );%+ double(dxy)*((m/S).^2));
        end
    end
end
