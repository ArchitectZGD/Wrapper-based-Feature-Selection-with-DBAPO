function [trIdx, teIdx] = makeOuterSplit(label, group, ho, seed)
% makeOuterSplit  Leakage-safe OUTER train/test split (per run).
% -------------------------------------------------------------------------
% Splits the FULL dataset into a TRAINING portion (returned to the optimiser)
% and a held-out TEST portion (used only once, in the final evaluation).
%
%   label : full label vector (n x 1)
%   group : sample->group ID vector; empty when the dataset has no grouping
%   ho    : fraction held out as the TEST set
%   seed  : per-run random seed (same seed -> same split; pass the SAME seed
%           to every algorithm so paired tests stay valid)
%
% Group structure: when `group` is non-empty, samples belonging to the same
% person/molecule are kept entirely on one side (fixes group leakage,
% Reviewer 4 point 2).
% -------------------------------------------------------------------------
    if ~isempty(group)
        HO = common.groupHoldout(group, ho, seed);      % group-level split
        trIdx = logical(HO.training);
        teIdx = logical(HO.test);
    else
        try
            cv = cvpartition(label, 'HoldOut', ho, 'Stratify', true);
        catch
            % rare classes can break stratified hold-out -> fall back
            cv = cvpartition(numel(label), 'HoldOut', ho);
        end
        trIdx = training(cv);
        teIdx = test(cv);
    end
end
