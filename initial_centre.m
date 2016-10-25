function [x_c , y_c ] = initial_centre(matrix, val)

[x,y]=find(matrix==val);
x_c=sum(x)/size(x,1);
y_c=sum(y)/size(y,1);

end

