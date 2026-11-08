function [] = generateOverView(max_Run, Algorithms)
    project_root = fileparts(fileparts(mfilename('fullpath')));
    for Algorithm_Name = Algorithms
        folderPath = fullfile(project_root, 'Data Result');
        files = dir(folderPath);
        file_pattern = string(max_Run) + '_summary.csv';
        good_performance = {};

        for i = 1:length(files)
            if files(i).isdir && ~strcmp(files(i).name,'.') && ~strcmp(files(i).name,'..')
                dspath = fullfile(folderPath, files(i).name);
                summaryfile_path = fullfile(dspath, file_pattern);
                if isfile(summaryfile_path)
                    T = readtable(summaryfile_path);
                    value = T{1,1}{1};
                    % Strip the trailing _<max_Run>.mat or _<max_Run>_<classifier>.mat from the file name
                    value = regexprep(value, '_\d+(_[A-Za-z]+)?\.mat$', '');
                    if value == Algorithm_Name
                        good_performance{end+1} = files(i).name;
                    end
                else
                    continue;
                end
            end
        end
        
        overview_dir = fullfile(project_root, 'Over View');
        if ~isfolder(overview_dir), mkdir(overview_dir); end
        outputpath = fullfile(overview_dir, string(Algorithm_Name) + '_' + string(max_Run) + '_overview.csv');
        writecell(good_performance(:), outputpath);
    end
end

