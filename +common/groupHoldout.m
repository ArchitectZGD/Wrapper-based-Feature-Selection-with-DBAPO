function HO = groupHoldout(group, ho, seed)
% Group-level HoldOut split: split the unique group IDs by (1-ho)/ho so all samples of a group stay on the same side
% Returns a struct with fields .training/.test (logical vectors), compatible with the same-named cvpartition fields
%
%   group : n×1 group ID of each sample (numeric or string)
%   ho    : test ratio (e.g., 0.2)
%   seed  : random seed (recommended 1000*didx + run so all algorithms share the same split for the same (dataset, run))
    g = group(:);

    if isnumeric(g)
        [~, ~, ic] = unique(g, 'stable');
    else
        [~, ~, ic] = unique(string(g), 'stable');
    end
    n_grp = max(ic);
    n_test = max(1, floor(n_grp * ho));
    test_grp = sort(randperm(n_grp, n_test));   % use the global rng (no seed set)

    is_test_grp = false(n_grp, 1);
    is_test_grp(test_grp) = true;

    test = is_test_grp(ic);
    HO = struct('training', ~test, 'test', test);
end
