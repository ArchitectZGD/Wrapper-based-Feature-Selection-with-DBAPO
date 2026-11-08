function error = jwrapperSVM(sFeat, label, HO)
% jwrapperSVM
% -------------------------------------------------------------------------
% Linear-kernel SVM (fitcecoc handles multiclass automatically) classification
% error on the INNER validation partition.
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
            foldErr(f) = svmFoldError(sFeat, label, training(HO, f), test(HO, f));
        end
        error = mean(foldErr);             % single value for hold-out; mean for k-fold
    else
        % Legacy struct HO: single train/valid fold.
        error = svmFoldError(sFeat, label, logical(HO.training), logical(HO.test));
    end
end


function foldErr = svmFoldError(sFeat, label, trMask, teMask)
% One fold of SVM training/validation. Trains only on trMask, predicts only
% on teMask; no information from teMask (or any outer test set) is used to fit.
    xtrain = sFeat(trMask, :);
    ytrain = label(trMask, :);
    xvalid = sFeat(teMask, :);
    yvalid = label(teMask, :);

    if numel(unique(ytrain)) == 1
        % Training fold has a single class (small subset) -> majority guess.
        pred = repmat(mode(ytrain), size(yvalid));
    else
        try
            t = templateSVM('KernelFunction', 'linear', 'Standardize', true);
            Model = fitcecoc(xtrain, ytrain, 'Learners', t);
            pred = predict(Model, xvalid);
        catch
            % Fit failed (e.g. degenerate fold) -> safe majority baseline.
            pred = repmat(mode(ytrain), size(yvalid));
        end
    end

    foldErr = 1 - (sum(yvalid(:) == pred(:)) / numel(yvalid));
end
