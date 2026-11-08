function error = jwrapperLR(sFeat, label, HO)
% jwrapperLR
% -------------------------------------------------------------------------
% Logistic regression (binary: fitclinear; multiclass: fitcecoc + templateLinear)
% classification error on the INNER validation partition.
%
% IMPORTANT (data leakage): follows the same leakage-free logic as
% jFitnessFunction / jwrapperKNN. HO must be an INNER cvpartition built solely
% from the TRAINING data (see buildInnerPartition in Main.m); the outer held-out
% TEST set is never visible here, so it cannot leak into the feature search.
%
% Supports BOTH inner schemes returned by buildInnerPartition:
%   - single stratified hold-out : HO.NumTestSets == 1
%   - stratified k-fold          : HO.NumTestSets  > 1  (error averaged
%                                  across folds, small-sample stability)
%
% For backward compatibility, a legacy struct HO with .training/.test fields
% (e.g. common.groupHoldout) is treated as a single fold.
% -------------------------------------------------------------------------

    if isa(HO, 'cvpartition')
        nSets   = HO.NumTestSets;          % 1 = hold-out, >1 = k-fold
        foldErr = zeros(nSets, 1);
        for f = 1:nSets
            foldErr(f) = lrFoldError(sFeat, label, training(HO, f), test(HO, f));
        end
        error = mean(foldErr);             % single value for hold-out; mean for k-fold
    else
        % Legacy struct HO: single train/valid fold.
        error = lrFoldError(sFeat, label, logical(HO.training), logical(HO.test));
    end
end


function foldErr = lrFoldError(sFeat, label, trMask, teMask)
% One fold of logistic-regression training/validation. Trains only on trMask,
% predicts only on teMask; no information from teMask is used to fit.
    xtrain = sFeat(trMask, :);
    ytrain = label(trMask, :);
    xvalid = sFeat(teMask, :);
    yvalid = label(teMask, :);

    if numel(unique(ytrain)) == 1
        % Training fold has a single class (small subset) -> majority guess.
        pred = repmat(mode(ytrain), size(yvalid));
    else
        try
            if numel(unique(ytrain)) == 2
                Model = fitclinear(xtrain, ytrain, 'Learner', 'logistic');
            else
                t = templateLinear('Learner', 'logistic');
                Model = fitcecoc(xtrain, ytrain, 'Learners', t);
            end
            pred = predict(Model, xvalid);
        catch
            % Fit failed (e.g. degenerate fold) -> safe majority baseline.
            pred = repmat(mode(ytrain), size(yvalid));
        end
    end

    foldErr = 1 - (sum(yvalid(:) == pred(:)) / numel(yvalid));
end
