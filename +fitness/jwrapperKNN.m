function error = jwrapperKNN(sFeat, label, HO)
% jwrapperKNN
% -------------------------------------------------------------------------
% KNN classification error on the INNER validation partition.
%
% IMPORTANT (data leakage): follows the same leakage-free logic as
% jFitnessFunction. HO must be an INNER cvpartition built solely from the
% TRAINING data (see buildInnerPartition in Main.m); the outer held-out TEST
% set is never visible here, so it cannot leak into the feature search.
%
% Supports BOTH inner schemes returned by buildInnerPartition:
%   - single stratified hold-out : HO.NumTestSets == 1
%   - stratified k-fold          : HO.NumTestSets  > 1  (error averaged
%                                  across folds, small-sample stability)
%
% For backward compatibility, a legacy struct HO with .training/.test fields
% (e.g. common.groupHoldout) is treated as a single fold.
% -------------------------------------------------------------------------

    k = 5;                                 % KNN neighbours (single point of control)

    if isa(HO, 'cvpartition')
        nSets   = HO.NumTestSets;          % 1 = hold-out, >1 = k-fold
        foldErr = zeros(nSets, 1);
        for f = 1:nSets
            foldErr(f) = knnFoldError(sFeat, label, training(HO, f), test(HO, f), k);
        end
        error = mean(foldErr);             % single value for hold-out; mean for k-fold
    else
        % Legacy struct HO: single train/valid fold.
        error = knnFoldError(sFeat, label, logical(HO.training), logical(HO.test), k);
    end
end


function foldErr = knnFoldError(sFeat, label, trMask, teMask, k)
% One fold of KNN training/validation. Trains only on trMask, predicts only
% on teMask; no information from teMask (or any outer test set) is used to fit.
    xtrain = sFeat(trMask, :);
    ytrain = label(trMask, :);
    xvalid = sFeat(teMask, :);
    yvalid = label(teMask, :);

    Model = fitcknn(xtrain, ytrain, 'NumNeighbors', k, 'Distance', 'euclidean', 'DistanceWeight', 'equal', 'BreakTies', 'smallest');
    pred  = predict(Model, xvalid);

    foldErr = 1 - (sum(yvalid(:) == pred(:)) / numel(yvalid));
end
