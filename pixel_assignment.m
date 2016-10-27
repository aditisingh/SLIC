function [labelled] = pixel_assignment(img_ht, img_wd, S, labelled, D, dxy, C_curr)

for i=1:size(unique(labelled),1)
    for j=max(C_curr(i,2)-S,1):min(C_curr(i,2)+S,img_ht)      %y_coord
        for k=max(C_curr(i,1)-S,1):min(C_curr(i,1)+S,img_wd)  %x-coord
            if D(j,k,i)==min(D(j,k,:))
                labelled(j,k)=i;
            end
        end
    end
end

end

