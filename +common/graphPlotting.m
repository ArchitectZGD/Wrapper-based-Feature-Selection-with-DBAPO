function [] = graphPlotting(Algorithm_Name, dataset_name, curve_All, ErrorCurve_All, featureSubsetSize_All, max_FEs, classifier_name, fitness_alpha)
    Plot_Accuracy_vs_FeatureSubsetSize = @common.Plot_Accuracy_vs_FeatureSubsetSize ;
    Algorithm_Name = Algorithm_Name + "-";
    % Include the classifier in the figure name (not for KNN) to avoid different classifiers overwriting each other
    if nargin >= 7 && ~isempty(classifier_name) && ~strcmp(upper(classifier_name), 'KNN')
        Algorithm_Name = Algorithm_Name + upper(classifier_name) + "-";
    end
    % Locate the project root via an absolute path (avoids Figure Result not being found when the working directory differs from the project root)
    [common_folder, ~, ~] = fileparts(mfilename('fullpath'));
    project_root = fileparts(common_folder);   % parent of +common = project root
    if nargin >= 8 && ~isempty(fitness_alpha) && abs(fitness_alpha - 0.99) > 1e-9
        figRoot = sprintf('Figure Result_a%.2f', fitness_alpha);   % robustness experiment: separate figure folder
    else
        figRoot = 'Figure Result';
    end
    folderPath = fullfile(project_root, figRoot, dataset_name);
    if ~isfolder(folderPath)
        mkdir(folderPath);
    end

    % Plot convergence curve (mean +/- std band over runs)
    figurename1 = [Algorithm_Name{1},'Convergence_DS'];
    figure('Name', figurename1, 'NumberTitle', 'off', 'Visible', 'off');
    plotMeanStd(1:max_FEs, curve_All);
    xlabel('Number of Fitness Evaluation');
    ylabel('Fitness Value');
    title(Algorithm_Name); grid on;
    saveas(gcf,(fullfile(folderPath, figurename1)),'tiff')

    % Plot error curve (mean +/- std band over runs)
    figurename2 = [Algorithm_Name{1}, 'Error_DS'];
    figure('Name', figurename2, 'NumberTitle', 'off', 'Visible', 'off');
    plotMeanStd(1:max_FEs, ErrorCurve_All);
    xlabel('Number of Fitness Evaluation');
    ylabel('Classification Error');
    title(Algorithm_Name);
    grid on;
    saveas(gcf,(fullfile(folderPath, figurename2)),'tiff')

    % Plot feature subset size curve (mean +/- std band over runs)
    figurename3 = [Algorithm_Name{1}, 'featureSubsetSize_DS'];
    figure('Name', figurename3, 'NumberTitle', 'off', 'Visible', 'off');
    plotMeanStd(1:max_FEs, featureSubsetSize_All);
    xlabel('Number of Fitness Evaluation');
    ylabel('Feature Subset Size');
    title(Algorithm_Name); grid on;
    saveas(gcf,(fullfile(folderPath, figurename3)),'tiff')

    % Plot Accuracy vs feature Subset Size curve
    figurename4 = [Algorithm_Name{1}, 'AccuracyvsfeatureSubsetSize_DS'];
    figure('Name', figurename4, 'NumberTitle', 'off', 'Visible', 'off');
    Plot_Accuracy_vs_FeatureSubsetSize(ErrorCurve_All, featureSubsetSize_All);
    saveas(gcf, fullfile(folderPath, figurename4), 'tiff');
end


function plotMeanStd(x, M)
% Plot mean trajectory (solid) with +/- one std band (shaded) across runs.
    m = mean(M, 1);
    s = std(M, 0, 1);
    x = x(:)'; m = m(:)'; s = s(:)';
    fill([x, fliplr(x)], [m+s, fliplr(m-s)], [0.85 0.85 0.85], ...
        'EdgeColor','none', 'FaceAlpha', 0.6); hold on;
    plot(x, m, 'LineWidth', 1.5); hold off;
end
