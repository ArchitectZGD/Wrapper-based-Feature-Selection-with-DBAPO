function M = computeMetrics(confMat, classOrder)
% computeMetrics  Aggregate classification metrics from a confusion matrix.
% -------------------------------------------------------------------------
%   confMat    : rows = true class, cols = predicted class (fixed order)
%   classOrder : (optional) class labels for the per-class table
%
% Returns struct M with:
%   accuracy, balancedAcc (= macro recall), macroPrec, macroRecall, macroF1,
%   microPrec, microRecall, microF1, mcc (multi-class, generalizes binary),
%   perClass (table: Class | Precision | Recall | F1).
%
% Convention: a class never predicted or never present in the test set gets
% precision/recall/F1 = 0 (its contribution to macro averages is 0).
% -------------------------------------------------------------------------
    if nargin < 2
        classOrder = (1:size(confMat, 1))';
    end
    classOrder = classOrder(:);
    nClass = size(confMat, 1);
    total  = sum(confMat(:));

    tp = diag(confMat);
    fp = sum(confMat, 1)' - tp;          % predicted as this class, truly other
    fn = sum(confMat, 2)  - tp;          % truly this class, predicted other

    prec = tp ./ (tp + fp);
    rec  = tp ./ (tp + fn);
    f1   = 2 .* prec .* rec ./ (prec + rec);
    prec(isnan(prec)) = 0;
    rec(isnan(rec))   = 0;
    f1(isnan(f1))     = 0;

    M.accuracy    = sum(tp) / total;
    M.balancedAcc = mean(rec);           % = macro recall
    M.macroPrec   = mean(prec);
    M.macroRecall = mean(rec);
    M.macroF1     = mean(f1);

    tpMicro = sum(tp); fpMicro = sum(fp); fnMicro = sum(fn);
    M.microPrec   = tpMicro / (tpMicro + fpMicro);
    M.microRecall = tpMicro / (tpMicro + fnMicro);
    M.microF1     = 2 * M.microPrec * M.microRecall / (M.microPrec + M.microRecall);
    if isnan(M.microF1), M.microF1 = 0; end

    % Multi-class MCC (generalizes the binary MCC; Gorodkin 2004):
    %   MCC = (c*N - sum(p.*t)) / sqrt((N^2 - sum(p.^2)) * (N^2 - sum(t.^2)))
    c = sum(tp);                       % number of correct predictions
    t = sum(confMat, 2);               % true-class counts  (row sums)
    p = sum(confMat, 1)';              % predicted-class counts (column sums)
    den = sqrt((total^2 - sum(p.^2)) * (total^2 - sum(t.^2)));
    if den > 0
        M.mcc = (c * total - sum(p .* t)) / den;
    else
        M.mcc = 0;
    end

    M.perClass = table(classOrder, prec, rec, f1, ...
        'VariableNames', {'Class','Precision','Recall','F1'});
end
