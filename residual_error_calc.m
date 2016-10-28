function [E ] = residual_error_calc(C_curr, C_prev)
    sum=0;
    for i=1:size(C_curr,1)
        sum=sum+(C_curr(i,1)-C_prev(i,1))^2+(C_curr(i,2)-C_prev(i,2))^2;
    end
    E=sum/size(C_curr,1);
end

