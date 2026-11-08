function [innerPart, scheme] = buildInnerPartition(ytr, ho_inner, groupTr, seed)
% buildInnerPartition  Leakage-free INNER validation partition.
% -------------------------------------------------------------------------
% Builds the partition used to score candidate feature subsets DURING the
% search. It is built ONLY from the TRAINING data (ytr / groupTr); the held-out
% TEST set is never passed in, so it cannot leak into the search.
%
% Group structure (Reviewer 4 point 2): when groupTr is provided, the inner
% split is performed at GROUP level, so samples from the same person/molecule
% remain entirely on one side of the inner train/validation split.
%
% Otherwise (no grouping):
%   - small datasets (min class < 5) switch to stratified k-fold for stability
%   - larger ones keep a single stratified hold-out for speed
%
% Output `innerPart` is either:
%   - a cvpartition object (sample-level; works with NumTestSets 1 or > 1), or
%   - a struct with .training/.test fields (group-level single hold-out).
% Both are handled by the wrappers.
% -------------------------------------------------------------------------
    if nargin < 4, seed = 0; end

    if nargin >= 3 && ~isempty(groupTr)
        % Group-level inner hold-out: same group stays whole.
        innerPart = common.groupHoldout(groupTr, ho_inner, seed);
        scheme    = 'group-level holdout';
        return;
    end

    classes  = unique(ytr);
    counts   = arrayfun(@(c) sum(ytr == c), classes);
    minClass = min(counts);

    if minClass >= 5
        innerPart = cvpartition(ytr, 'HoldOut', ho_inner, 'Stratify', true);
        scheme    = 'stratified holdout';
    elseif minClass >= 2
        nFolds    = min(5, minClass);          % every class must fill every fold
        innerPart = cvpartition(ytr, 'KFold', nFolds, 'Stratify', true);
        scheme    = sprintf('stratified %d-fold', nFolds);
    else
        % A class with a single training sample cannot be stratified at all;
        % fall back to a plain hold-out (visible, not silently mishandled).
        innerPart = cvpartition(numel(ytr), 'HoldOut', ho_inner);
        scheme    = 'non-stratified holdout (degenerate)';
    end
end
