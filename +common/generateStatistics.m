function [] = generateStatistics(folder_path, max_Run)
    % Match *_<max_Run>.mat or *_<max_Run>_<classifier>.mat (e.g., DBAPO_C3_20_SVM.mat)
    mat_files = dir(fullfile(folder_path, '*.mat'));
    pat = ['_' num2str(max_Run) '(_[A-Za-z]+)?\.mat$'];
    keep = false(numel(mat_files), 1);
    for k = 1:numel(mat_files)
        keep(k) = ~isempty(regexp(mat_files(k).name, pat, 'once'));
    end
    mat_files = mat_files(keep);

    % Special logic fields (Mean_TestAcc is the headline metric of three-layer validation, placed first)
    priority_vars = {'Mean_TestAcc','Mean_BalancedAcc','Mean_MacroF1','Mean_MacroRecall','Mean_MacroPrec','Mean_Acc','Ave_Fitness','Ave_Nf','total_feature'};
    exclude_vars = {'curve_All','ErrorCurve_All','featureSubsetSize_All','Sf_All','Nf_All','All_Fitness', ...
                    'ConfMat_All','Metrics_All','TestIdx_All','InnerScheme_All','ClassOrder', ...
                    'TestAcc_All','BalancedAcc_All','MacroF1_All','MacroPrec_All','MacroRecall_All', ...
                    'group'};
    
    % Temporary cell to store table data
    data = {};
    all_vars = {};  % collect all parameter names (for dynamic table construction later)
    
    for i = 1:length(mat_files)
        file_path = fullfile(mat_files(i).folder, mat_files(i).name);
        S = load(file_path);
        var_names = fieldnames(S);
    
        % Remove excluded fields
        var_names = setdiff(var_names, exclude_vars);
    
        % Record all variable names
        all_vars = union(all_vars, var_names);
    
        % Save values, put file name first
        row.FileName = string(mat_files(i).name);
    
        % Iterate over variables in current file
        for v = var_names'
            name = v{1};
            value = S.(name);
    
            % Numeric arrays take first value (can use mean, etc.); keep char/string (e.g., classifier) as-is
            if isnumeric(value) && ~isscalar(value)
                value = value(1);
            end
            row.(name) = value;
        end
    
        data{end+1} = row;
    end
    
    % Construct final column order
    real_vars = setdiff(all_vars, exclude_vars);
    
    other = setdiff(real_vars, priority_vars, 'stable');
    other = other(:)'; 
    ordered_vars = [{'FileName'}, priority_vars, other];
    
    % Build table
    T = table();
    
    for i = 1:length(data)
        r = data{i};
    
        % Insert each row in column order
        row_data = cell(1, length(ordered_vars));
        for j = 1:length(ordered_vars)
            v = ordered_vars{j};
            if isfield(r, v)
                row_data{j} = r.(v);
            elseif strcmp(v, 'classifier')
                row_data{j} = '';   % old result files lack classifier field; fill with empty string to avoid type conflicts
            else
                row_data{j} = NaN;
            end
        end
    
        T = [T; cell2table(row_data, 'VariableNames', ordered_vars)];
    end

    T = sortrows(T, {'Mean_TestAcc','Mean_Acc','Ave_Fitness','Ave_Nf'}, {'descend','descend','ascend','ascend'});
    disp('-----------------------');
    disp(T);
    disp('-----------------------');
    csvname = string(max_Run) + '_summary.csv';
    writetable(T, fullfile(folder_path, csvname));
end 