function [featTr, featTe] = preprocessData(featTr, featTe)
% preprocessData  Leakage-free imputation of missing values.
% -------------------------------------------------------------------------
% Treats NaN and +/-Inf as missing. For each feature column, the fill value
% is the MEDIAN computed from the TRAINING partition ONLY, then applied to
% BOTH the training and test partitions. A column with no valid training
% values is filled with 0.
%
% MUST be called AFTER the outer train/test split, so the test set never
% contributes to the imputation (fixes preprocessing leakage, Reviewer 4
% point 1: "Imputation ... must be fitted within the training partition").
% -------------------------------------------------------------------------
    featTr = double(featTr);
    featTe = double(featTe);
    nCol = size(featTr, 2);

    for j = 1:nCol
        colTr = featTr(:, j);
        missTr = isnan(colTr) | isinf(colTr);

        % Fill value from TRAINING only.
        validTr = colTr(~missTr);
        if isempty(validTr)
            fillVal = 0;
        else
            fillVal = median(validTr);
        end

        if any(missTr)
            colTr(missTr) = fillVal;
            featTr(:, j) = colTr;
        end

        % Apply the SAME (training-derived) value to the test partition.
        colTe = featTe(:, j);
        missTe = isnan(colTe) | isinf(colTe);
        if any(missTe)
            colTe(missTe) = fillVal;
            featTe(:, j) = colTe;
        end
    end
end
