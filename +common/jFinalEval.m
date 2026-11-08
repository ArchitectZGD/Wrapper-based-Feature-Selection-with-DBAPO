function [confMat, testAcc, classOrder, metrics] = jFinalEval(featTr, labelTr, featTe, labelTe, Sf, classifierName)
% jFinalEval  Leakage-free final evaluation on the held-out TEST set.
% -------------------------------------------------------------------------
% Trains the classifier on the FULL training data using the selected
% features, then scores ONCE on the held-out test set. The test set has
% never been seen by the search, so this is an honest generalization estimate.
%
%   classifierName : 'KNN' (default) | 'SVM' | 'RF' | 'LR' | 'MLP'
%                    Must match the classifier used during the search.
%
%   confMat    : rows = true class, cols = predicted class (fixed order)
%   testAcc    : held-out test-set accuracy (NaN if no feature selected)
%   classOrder : fixed class list (same-size matrix every run)
% -------------------------------------------------------------------------
    if nargin < 6 || isempty(classifierName)
        classifierName = 'KNN';
    end
    classifierName = upper(classifierName);

    labelTr = labelTr(:);
    labelTe = labelTe(:);
    classOrder = unique([labelTr; labelTe]);

    if isempty(Sf)                       % guard: no feature selected
        confMat = zeros(numel(classOrder));
        testAcc = NaN;
        metrics = common.computeMetrics(confMat, classOrder);
        return;
    end

    Xtr = featTr(:, Sf);
    Xte = featTe(:, Sf);
    ytr = labelTr;
    yte = labelTe;

    if numel(unique(ytr)) == 1
        pred = repmat(mode(ytr), size(yte));     % single-class -> majority guess
    else
        try
            switch classifierName
                case 'SVM'
                    t = templateSVM('KernelFunction', 'linear', 'Standardize', true);
                    mdl = fitcecoc(Xtr, ytr, 'Learners', t);
                    pred = predict(mdl, Xte);
                case 'RF'
                    mdl = TreeBagger(30, Xtr, ytr, 'Method', 'classification', 'OOBPrediction', 'off');
                    pred = predict(mdl, Xte);
                    if iscell(pred), pred = str2double(pred); end
                case 'LR'
                    if numel(unique(ytr)) == 2
                        mdl = fitclinear(Xtr, ytr, 'Learner', 'logistic');
                    else
                        t = templateLinear('Learner', 'logistic');
                        mdl = fitcecoc(Xtr, ytr, 'Learners', t);
                    end
                    pred = predict(mdl, Xte);
                case 'MLP'
                    classLabels = unique(ytr);
                    [~, ytrIdx] = ismember(ytr, classLabels);
                    net = patternnet(10);
                    net.trainParam.showWindow = false;
                    net.trainParam.showCommandLine = false;
                    net = train(net, Xtr', full(ind2vec(ytrIdx')));
                    out = net(Xte');
                    [~, predIdx] = max(out, [], 1);
                    pred = classLabels(predIdx)';
                otherwise   % 'KNN'
                    mdl = fitcknn(Xtr, ytr, 'NumNeighbors', 5, 'Distance', 'euclidean', 'DistanceWeight', 'equal', 'BreakTies', 'smallest');
                    pred = predict(mdl, Xte);
            end
            pred = pred(:);
        catch
            pred = repmat(mode(ytr), size(yte));
        end
    end

    confMat = confusionmat(yte, pred, 'Order', classOrder);
    testAcc = sum(pred == yte) / numel(yte);
    metrics = common.computeMetrics(confMat, classOrder);
end
