function [labelled] = pixel_assignment(img_ht, img_wd, S, labelled, D, dxy)

for j=1:img_ht
    for k=1:img_wd
        %find the pixels in range 2S of the pixel
        idx=find(sqrt(dxy(j,k,:))<=S);
        %find lowest weight
        if(size(find(D(j,k,idx)==min(D(j,k,idx))))==1)
            labelled(j,k)=find(D(j,k,idx)==min(D(j,k,idx)));
        else
            vals=find(D(j,k,idx)==min(D(j,k,idx)));
            labelled(j,k)=idx(vals(1));   %allocate to anyone %think more about this
        end
    end
end

end

