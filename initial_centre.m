function [x_c , y_c ] = initial_centre(matrix, val)

[y,x]=find(matrix==val);
x_c=round(sum(x)/size(x,1));
y_c=round(sum(y)/size(y,1));

end

