clc; clear; close all;
point = [10457 9156];
dist1 = 1387;
dist2 = 856;

% feature Table
table = readtable('../Downloads/Table_inj.xlsx');
%[k(:,1),k(:,2),k(:,3)] = find (label);                                      % finding (x,y) position and index of each pixel belonging to a cell 
%k2 = unique(k(:,3), 'stable');

cnt=1;

for i=3:size(table,1)
    if table.centroid_x(i) >= point(1) 
        if table.centroid_x(i) <= (point(1)+dist1 -1)
            if table.centroid_y(i) >= point(2)
                if table.centroid_y(i) <= (point(2) +dist2-1)
                    featureTable(cnt,:) = table(table.ID(i)-2,:);
                    featureTable.centroid_x(cnt)=featureTable.centroid_x(cnt) - point1(1) + 1; 
                    featureTable.centroid_y(cnt)=featureTable.centroid_y(cnt) - point1(2) + 1; 
                    cnt=cnt+1, i
                end
            end
        end
    end
end

