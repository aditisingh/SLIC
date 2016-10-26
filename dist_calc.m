function [D, dxy] = dist_calc(labelled,m, S, C, stack, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, ...
lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19, lambda_20, lambda_21, lambda_22, lambda_23, lambda_24)

[img_ht,img_wd]=size(labelled);

D=zeros(img_ht, img_wd, size(unique(labelled),1)); %distance for each pixel from each centre
dxy=zeros(img_ht,img_wd, size(unique(labelled),1));

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
            
            dxy(j,k,i)=((k-C(i,1)).^2+(j-C(i,2)).^2);
            
            D(j,k,i)=sqrt(double(d1)*(lambda_1).^2 + double(d2)*(lambda_2).^2 + double(d3)*(lambda_3).^2 + double(d4)*(lambda_4).^2 + double(d5)*(lambda_5).^2 ... 
                + double(d6)*(lambda_6).^2 + double(d7)*(lambda_7).^2 + double(d8)*(lambda_8).^2 + double(d9)*(lambda_9).^2 + double(d10)*(lambda_10).^2 ...
                + double(d11)*(lambda_11).^2 + double(d12)*(lambda_12).^2 + double(d13)*(lambda_13).^2 + double(d14)*(lambda_14).^2 + double(d15)*(lambda_15).^2 ...
                + double(d16)*(lambda_16).^2 + double(d17)*(lambda_17).^2 + double(d18)*(lambda_18).^2 + double(d19)*(lambda_19).^2 + double(d20)*(lambda_20).^2 ...
                + double(d21)*(lambda_21).^2 + double(d22)*(lambda_22).^2 + double(d23)*(lambda_23).^2 + double(d24)*(lambda_24).^2 + double(dxy(j,k,i)*((m/S).^2)));
        end
    end
end

end

