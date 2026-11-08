function [X] = random_select_mapping_linear(X, Total_Feature, thres)    
    
    for d = 1: Total_Feature
        if X(d) > thres
            X(d) = 1; 
        else
            X(d) = 0;
        end
    end
    
end

