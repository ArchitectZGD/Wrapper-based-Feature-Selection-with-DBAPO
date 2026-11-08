function error = jwrapperMLP(sFeat, label, HO)
% jwrapperMLP
% -------------------------------------------------------------------------
% MLP (feedforward patternnet) classification error on the INNER validation
% partition.
%
% IMPORTANT (data leakage): follows the same leakage-free logic as
% jFitnessFunction / jwrapperKNN. HO must be an INNER cvpartition built solely
% from the TRAINING data (see buildInnerPartition in Main.m); the outer held-out
% TEST set is never visible here, so it cannot leak into the feature search.
% The class-label mapping is learned from the TRAINING fold only.
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
            foldErr(f) = mlpFoldError(sFeat, label, training(HO, f), test(HO, f));
        end
        error = mean(foldErr);             % single value for hold-out; mean for k-fold
    else
        % Legacy struct HO: single train/valid fold.
        error = mlpFoldError(sFeat, label, logical(HO.training), logical(HO.test));
    end
end


function foldErr = mlpFoldError(sFeat, label, trMask, teMask)
% One fold of MLP training/validation. Trains only on trMask, predicts only
% on teMask; the class mapping is derived from trMask only.
    xtrain = sFeat(trMask, :);
    ytrain = label(trMask, :);
    xvalid = sFeat(teMask, :);
    yvalid = label(teMask, :);

    % Class mapping learned ONLY from the training fold (leakage-free).
    classLabels = unique(ytrain);

    if numel(classLabels) == 1
        % Training fold has a single class (small subset) -> majority guess.
        pred = repmat(mode(ytrain), size(yvalid));
    else
        try
            [~, ytrain_idx] = ismember(ytrain, classLabels);

            % Simple feedforward net, 1 hidden layer (10 units).
            net = patternnet(10);
            net.trainParam.showWindow = false;       % no training GUI
            net.trainParam.showCommandLine = false;

            ytrain_onehot = full(ind2vec(ytrain_idx'));
            net = train(net, xtrain', ytrain_onehot);

            yvalid_pred_onehot = net(xvalid');
            [~, pred_idx] = max(yvalid_pred_onehot, [], 1);
            pred = classLabels(pred_idx)';
        catch
            % Fit failed (e.g. degenerate fold) -> safe majority baseline.
            pred = repmat(mode(ytrain), size(yvalid));
        end
    end

    foldErr = 1 - (sum(yvalid(:) == pred(:)) / numel(yvalid));
end
