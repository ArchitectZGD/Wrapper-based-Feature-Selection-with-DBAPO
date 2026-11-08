function Plot_Accuracy_vs_FeatureSubsetSize(ErrorCurve_All, FeatureSubsetSize_All)
    accData = 1 - mean(ErrorCurve_All);                   % Accuracy = 1 - mean(Error)
    fssData = floor(mean(FeatureSubsetSize_All));         % Feature Subset Size (rounded down)
    x = 1:length(accData);

    % Compute min, max, and padded limits for Accuracy
    accMin = min(accData);
    accMax = max(accData);
    accRange = accMax - accMin;

    if accRange == 0
        accRange = 0.1;
    end

    paddingRatio = 0.05;  % 5% padding
    accLower = accMin - paddingRatio * accRange;
    accUpper = accMax + paddingRatio * accRange;

    % Compute min, max, and padded limits for Feature Subset Size
    fssMin = min(fssData);
    fssMax = max(fssData);
    fssRange = fssMax - fssMin;

    if fssRange == 0
        fssRange = 0.1;
    end
    fssLower = fssMin - paddingRatio * fssRange;
    fssUpper = fssMax + paddingRatio * fssRange;

    % Plot Accuracy on the Left y-axis
    yyaxis left
    plot(x, accData, '-b', 'LineWidth', 2);
    ylabel('Accuracy');
    ylim([accLower, accUpper]);  % Apply padded limits for Accuracy

    % Plot Feature Subset Size on the Right y-axis
    yyaxis right
    plot(x, fssData, '--r', 'LineWidth', 2);
    ylabel('Feature Subset Size');
    ylim([fssLower, fssUpper]);  % Apply padded limits for Feature Subset Size

    % Common x-axis and other plot settings
    xlabel('Fitness Evaluation Number');
    grid on;
    legend('Accuracy','Feature Subset Size','Location','southeast');

    % Optionally add a title if needed:
    % title('Variation of Accuracy and Feature Subset Size with Fitness Evaluations');
end